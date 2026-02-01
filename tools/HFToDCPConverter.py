#!/usr/bin/env python
# Copyright (c) 2024, Megatron Contributors. All rights reserved.
#
# This script converts HuggingFace checkpoint to Megatron DCP format.
# It is the reverse operation of DCPConverter.py

import argparse
import gc
import json
import logging
import os
import re

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import save
from torch.distributed.checkpoint.state_dict_saver import save_state_dict
from torch.distributed.checkpoint import FileSystemWriter


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def inverse_transform_qkv_weight(args, q_proj, k_proj, v_proj):
    """
    Convert separate Q, K, V projections back to Megatron's interleaved QKV format.
    
    Megatron format for each query group:
        [q_heads (kv_channels each), q_heads (kv_channels each) for RoPE, k (kv_channels), v (kv_channels)]
    
    This reverses the transform_qkv_weight function from DCPConverter.py.
    """
    each_kv_size = args.kv_channels
    each_q_size = args.kv_channels * args.num_attention_heads // args.num_query_groups
    
    # Split Q into groups
    num_heads_per_group = args.num_attention_heads // args.num_query_groups
    
    # q_proj shape: [num_attention_heads * kv_channels, hidden_size]
    # k_proj shape: [num_query_groups * kv_channels, hidden_size]
    # v_proj shape: [num_query_groups * kv_channels, hidden_size]
    
    qs = torch.chunk(q_proj, args.num_attention_heads, dim=0)
    ks = torch.chunk(k_proj, args.num_query_groups, dim=0)
    vs = torch.chunk(v_proj, args.num_query_groups, dim=0)
    
    result = []
    head_idx = 0
    for group_idx in range(args.num_query_groups):
        # For each group, we need: q_heads (first half), q_heads (second half for RoPE), k, v
        group_qs_first = []
        group_qs_second = []
        
        for _ in range(num_heads_per_group):
            # Split each head into first and second half (for RoPE)
            # Actually in DCPConverter, it interleaves them: q, g, q, g, ...
            # So we need to reverse: pair them back
            group_qs_first.append(qs[head_idx])
            head_idx += 1
        
        # In the original transform: q and g are interleaved
        # q_split = q.split(args.kv_channels, dim=0)
        # g_split = g.split(args.kv_channels, dim=0)
        # for _q, _g in zip(q_split, g_split):
        #     qs.append(_q)
        #     qs.append(_g)
        
        # So the qs array has: [q0, g0, q1, g1, ...]
        # We need to reconstruct the original q and g from this
        
    # Let me re-analyze the original transform more carefully
    # In transform_qkv_weight:
    # - weight is [num_query_groups * (each_q_size + each_q_size + each_kv_size + each_kv_size), hidden]
    # - For each query group chunk:
    #   - Split into q, g, k, v
    #   - q and g are each `each_q_size` = kv_channels * num_heads_per_group
    #   - k and v are each `each_kv_size` = kv_channels
    #   - q_split and g_split are split by kv_channels (num_heads_per_group chunks each)
    #   - Interleave q_split[i] and g_split[i]
    
    # Actually the output is:
    # all_q = concat of [q0_0, g0_0, q0_1, g0_1, ..., q1_0, g1_0, ...] for all groups
    # all_k = concat of [k0, k1, ...] for all groups  
    # all_v = concat of [v0, v1, ...] for all groups
    
    # So to reverse, we need to:
    # 1. Un-interleave all_q to get q and g for each group
    # 2. Reconstruct the original qkv weight
    
    return _inverse_transform_qkv_impl(args, q_proj, k_proj, v_proj)


def _inverse_transform_qkv_impl(args, q_proj, k_proj, v_proj):
    """
    Reverse the QKV transformation from DCPConverter.
    
    Original transform_qkv_weight (Megatron -> HF):
        For each query group, Megatron has [q, g, k, v] where:
        - q, g: each have shape (num_heads_per_group * kv_channels, hidden_size)
                q and g are split by kv_channels and interleaved into all_q
        - k, v: each have shape (kv_channels, hidden_size)
        
        Output: all_q (interleaved q,g), all_k, all_v
    
    This function reverses it: HF [q_proj, k_proj, v_proj] -> Megatron [q*g*k*v per group]
    
    Note: This implementation handles BOTH:
    1. Standard HF format (no q/g split, q_proj = num_heads * kv_channels)
    2. Gauss/DeepSeek-style format with interleaved q/g (q_proj = 2 * num_heads * kv_channels)
    """
    kv_channels = args.kv_channels
    num_query_groups = args.num_query_groups
    num_attention_heads = args.num_attention_heads
    num_heads_per_group = num_attention_heads // num_query_groups
    hidden_size = q_proj.shape[1]
    
    each_q_size = kv_channels * num_heads_per_group
    each_kv_size = kv_channels
    
    # Detect if q_proj contains interleaved q/g (size = 2 * num_heads * kv_channels)
    # or standard format (size = num_heads * kv_channels)
    expected_standard_q_size = num_attention_heads * kv_channels
    expected_interleaved_q_size = 2 * num_attention_heads * kv_channels
    actual_q_size = q_proj.shape[0]
    
    if actual_q_size == expected_interleaved_q_size:
        # Interleaved q/g format (Gauss/DeepSeek style)
        return _inverse_transform_qkv_interleaved(args, q_proj, k_proj, v_proj)
    else:
        # Standard HF format
        return _inverse_transform_qkv_standard(args, q_proj, k_proj, v_proj)


def _inverse_transform_qkv_standard(args, q_proj, k_proj, v_proj):
    """
    Inverse transform for standard HF format without q/g interleaving.
    
    Standard Megatron QKV format per group: [Q_heads, K, V]
    """
    kv_channels = args.kv_channels
    num_query_groups = args.num_query_groups
    num_attention_heads = args.num_attention_heads
    num_heads_per_group = num_attention_heads // num_query_groups
    
    qkv_chunks = []
    
    for group_idx in range(num_query_groups):
        # Get Q heads for this group
        q_start = group_idx * num_heads_per_group * kv_channels
        q_end = (group_idx + 1) * num_heads_per_group * kv_channels
        q_group = q_proj[q_start:q_end]
        
        # Get K and V for this group
        k_group = k_proj[group_idx * kv_channels:(group_idx + 1) * kv_channels]
        v_group = v_proj[group_idx * kv_channels:(group_idx + 1) * kv_channels]
        
        # Megatron format: Q, K, V concatenated per group
        qkv_chunks.append(torch.cat([q_group, k_group, v_group], dim=0))
    
    return torch.cat(qkv_chunks, dim=0)


def _inverse_transform_qkv_interleaved(args, q_proj, k_proj, v_proj):
    """
    Inverse transform for Gauss/DeepSeek style with interleaved q/g.
    
    In this format, transform_qkv_weight outputs:
        all_q = [q0_0, g0_0, q0_1, g0_1, ..., q0_{n-1}, g0_{n-1}, q1_0, g1_0, ...]
        where each q_i and g_i is kv_channels in size
    
    Original Megatron format per group: [q, g, k, v]
        - q, g: each (num_heads_per_group * kv_channels, hidden_size)
        - k, v: each (kv_channels, hidden_size)
    """
    kv_channels = args.kv_channels
    num_query_groups = args.num_query_groups
    num_attention_heads = args.num_attention_heads
    num_heads_per_group = num_attention_heads // num_query_groups
    
    # Split q_proj into kv_channels sized chunks
    q_chunks = torch.chunk(q_proj, q_proj.shape[0] // kv_channels, dim=0)
    
    qkv_chunks = []
    chunk_idx = 0
    
    for group_idx in range(num_query_groups):
        # Reconstruct q and g from interleaved chunks
        q_parts = []
        g_parts = []
        
        for head_idx in range(num_heads_per_group):
            q_parts.append(q_chunks[chunk_idx])
            g_parts.append(q_chunks[chunk_idx + 1])
            chunk_idx += 2
        
        q_group = torch.cat(q_parts, dim=0)
        g_group = torch.cat(g_parts, dim=0)
        
        # Get K and V for this group
        k_group = k_proj[group_idx * kv_channels:(group_idx + 1) * kv_channels]
        v_group = v_proj[group_idx * kv_channels:(group_idx + 1) * kv_channels]
        
        # Megatron format: [q, g, k, v] per group
        qkv_chunks.append(torch.cat([q_group, g_group, k_group, v_group], dim=0))
    
    return torch.cat(qkv_chunks, dim=0)


def merge_gate_up_proj(gate_proj, up_proj):
    """
    Merge gate_proj and up_proj back to Megatron's linear_fc1 format.
    Megatron format: [gate; up] concatenated
    """
    return torch.cat([gate_proj, up_proj], dim=0)


class HFKeyMapper:
    """
    Maps HuggingFace checkpoint keys to Megatron DCP format.
    This is the inverse of DCPKeyMapper.
    """
    hf_layer_prefix = 'model.layers'
    megatron_layer_prefix = 'decoder.layers'
    
    # Reverse of DCPKeyMapper.key_map
    key_map = {
        'model.embed_tokens.weight': 'embedding.word_embeddings.weight',
        'model.norm.weight': 'decoder.final_layernorm.weight',
        'lm_head.weight': 'output_layer.weight',
        # Layer-level mappings (will be prefixed with layer index)
        'input_layernorm.weight': 'self_attention.linear_qkv.layer_norm_weight',
        'post_attention_layernorm.weight': 'pre_mlp_layernorm.weight',
        'mlp.gate.e_score_correction_bias': 'mlp.router.expert_bias',
        'mlp.gate.weight': 'mlp.router.weight',
        'self_attn.q_norm.weight': 'self_attention.q_layernorm.weight',
        'self_attn.k_norm.weight': 'self_attention.k_layernorm.weight',
        'self_attn.o_proj.weight': 'self_attention.linear_proj.weight',
        'mlp.down_proj.weight': 'mlp.linear_fc2.weight',
    }
    
    # Keys that need special handling (QKV merge, gate/up merge)
    qkv_keys = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight']
    mlp_keys = ['mlp.gate_proj.weight', 'mlp.up_proj.weight']
    
    layer_pattern = re.compile(r'model\.layers\.(\d+)\.(.+)')
    
    @classmethod
    def map_key(cls, hf_key):
        """Map a single HF key to Megatron key (without handling merges)."""
        m = re.match(cls.layer_pattern, hf_key)
        if m:
            layer_idx, subkey = m.groups()
            if subkey in cls.key_map:
                return f'{cls.megatron_layer_prefix}.{layer_idx}.{cls.key_map[subkey]}'
        elif hf_key in cls.key_map:
            return cls.key_map[hf_key]
        return None


class HFToDCPConverter:
    def __init__(
        self,
        hf_ckpt_path: str,
        dst: str,
        megatron_args: dict,
    ):
        """
        Args:
            hf_ckpt_path: Path to HuggingFace checkpoint directory
            dst: Destination path for Megatron DCP checkpoint
            megatron_args: Dictionary of Megatron arguments needed for conversion
        """
        self.hf_ckpt_path = hf_ckpt_path
        self.dst = dst
        self.megatron_args = megatron_args
        
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        # Build args namespace
        from argparse import Namespace
        self.args = Namespace(**megatron_args)
    
    def load_hf_weights(self):
        """Load weights from HuggingFace checkpoint."""
        state_dict = {}
        
        # Check for safetensors index
        index_file = os.path.join(self.hf_ckpt_path, 'model.safetensors.index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            # Load from sharded safetensors
            loaded_files = set()
            for key, filename in index['weight_map'].items():
                if filename not in loaded_files:
                    filepath = os.path.join(self.hf_ckpt_path, filename)
                    with safe_open(filepath, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            state_dict[k] = f.get_tensor(k)
                    loaded_files.add(filename)
        else:
            # Try single safetensors file
            single_file = os.path.join(self.hf_ckpt_path, 'model.safetensors')
            if os.path.exists(single_file):
                with safe_open(single_file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            else:
                # Try pytorch format
                for filename in os.listdir(self.hf_ckpt_path):
                    if filename.endswith('.bin') or filename.endswith('.pt'):
                        filepath = os.path.join(self.hf_ckpt_path, filename)
                        loaded = torch.load(filepath, map_location='cpu', weights_only=True)
                        state_dict.update(loaded)
        
        return state_dict
    
    def load_hf_config(self):
        """Load HuggingFace config.json."""
        config_path = os.path.join(self.hf_ckpt_path, 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def convert(self):
        """Main conversion function."""
        logging.info(f"Loading HuggingFace checkpoint from {self.hf_ckpt_path}")
        
        # Load HF config
        hf_config = self.load_hf_config()
        self._update_args_from_config(hf_config)
        
        # Load HF weights
        hf_state_dict = self.load_hf_weights()
        logging.info(f"Loaded {len(hf_state_dict)} tensors from HF checkpoint")
        
        # Convert to Megatron format
        megatron_state_dict = self._convert_state_dict(hf_state_dict)
        logging.info(f"Converted to {len(megatron_state_dict)} Megatron tensors")
        
        # Save as DCP format
        self._save_dcp(megatron_state_dict)
        logging.info(f"Saved Megatron DCP checkpoint to {self.dst}")
        
        # Save common.pt with args
        self._save_common()
        
        return megatron_state_dict
    
    def _update_args_from_config(self, hf_config):
        """Update args from HF config."""
        if not hasattr(self.args, 'num_layers'):
            self.args.num_layers = hf_config.get('num_hidden_layers', 32)
        if not hasattr(self.args, 'hidden_size'):
            self.args.hidden_size = hf_config.get('hidden_size', 4096)
        if not hasattr(self.args, 'num_attention_heads'):
            self.args.num_attention_heads = hf_config.get('num_attention_heads', 32)
        if not hasattr(self.args, 'num_query_groups'):
            self.args.num_query_groups = hf_config.get('num_key_value_heads', self.args.num_attention_heads)
        if not hasattr(self.args, 'kv_channels'):
            self.args.kv_channels = self.args.hidden_size // self.args.num_attention_heads
        if not hasattr(self.args, 'vocab_size'):
            self.args.vocab_size = hf_config.get('vocab_size', 32000)
        if not hasattr(self.args, 'ffn_hidden_size'):
            self.args.ffn_hidden_size = hf_config.get('intermediate_size', self.args.hidden_size * 4)
        
        # MoE config
        if not hasattr(self.args, 'num_experts'):
            self.args.num_experts = hf_config.get('num_local_experts', 0)
        if not hasattr(self.args, 'moe_layer_freq'):
            # Convert HF's first_k_dense_replace to Megatron's moe_layer_freq list
            # first_k_dense_replace = N means first N layers are dense, rest are MoE
            first_k_dense = hf_config.get('first_k_dense_replace', 0)
            if first_k_dense > 0 and self.args.num_experts > 0:
                # [0, 0, ..., 1, 1, ...] - first_k_dense zeros, then ones
                self.args.moe_layer_freq = [0] * first_k_dense + [1] * (self.args.num_layers - first_k_dense)
            elif self.args.num_experts > 0:
                # All layers are MoE
                self.args.moe_layer_freq = [1] * self.args.num_layers
            else:
                # No MoE
                self.args.moe_layer_freq = [0] * self.args.num_layers
    
    def _convert_state_dict(self, hf_state_dict):
        """Convert HF state dict to Megatron format."""
        megatron_state_dict = {}
        
        # Group keys by layer
        layer_keys = {}
        other_keys = {}
        
        for key in hf_state_dict.keys():
            m = re.match(r'model\.layers\.(\d+)\.(.+)', key)
            if m:
                layer_idx = int(m.group(1))
                subkey = m.group(2)
                if layer_idx not in layer_keys:
                    layer_keys[layer_idx] = {}
                layer_keys[layer_idx][subkey] = key
            else:
                other_keys[key] = key
        
        # Convert non-layer keys
        for hf_key in other_keys.keys():
            megatron_key = self._map_simple_key(hf_key)
            if megatron_key:
                megatron_state_dict[megatron_key] = hf_state_dict[hf_key]
        
        # Convert layer keys
        for layer_idx in sorted(layer_keys.keys()):
            layer_dict = layer_keys[layer_idx]
            self._convert_layer(layer_idx, layer_dict, hf_state_dict, megatron_state_dict)
        
        return megatron_state_dict
    
    def _map_simple_key(self, hf_key):
        """Map simple (non-layer) HF key to Megatron key."""
        simple_map = {
            'model.embed_tokens.weight': 'embedding.word_embeddings.weight',
            'model.norm.weight': 'decoder.final_layernorm.weight',
            'lm_head.weight': 'output_layer.weight',
        }
        return simple_map.get(hf_key)
    
    def _convert_layer(self, layer_idx, layer_dict, hf_state_dict, megatron_state_dict):
        """Convert a single transformer layer."""
        prefix = f'decoder.layers.{layer_idx}'
        hf_prefix = f'model.layers.{layer_idx}'
        
        # Handle QKV merge
        if 'self_attn.q_proj.weight' in layer_dict:
            q_proj = hf_state_dict[f'{hf_prefix}.self_attn.q_proj.weight']
            k_proj = hf_state_dict[f'{hf_prefix}.self_attn.k_proj.weight']
            v_proj = hf_state_dict[f'{hf_prefix}.self_attn.v_proj.weight']
            qkv = _inverse_transform_qkv_impl(self.args, q_proj, k_proj, v_proj)
            megatron_state_dict[f'{prefix}.self_attention.linear_qkv.weight'] = qkv
        
        # Handle gate/up merge
        if 'mlp.gate_proj.weight' in layer_dict:
            gate = hf_state_dict[f'{hf_prefix}.mlp.gate_proj.weight']
            up = hf_state_dict[f'{hf_prefix}.mlp.up_proj.weight']
            fc1 = merge_gate_up_proj(gate, up)
            megatron_state_dict[f'{prefix}.mlp.linear_fc1.weight'] = fc1
        
        # Handle simple layer mappings
        simple_layer_map = {
            'input_layernorm.weight': 'self_attention.linear_qkv.layer_norm_weight',
            'post_attention_layernorm.weight': 'pre_mlp_layernorm.weight',
            'self_attn.o_proj.weight': 'self_attention.linear_proj.weight',
            'mlp.down_proj.weight': 'mlp.linear_fc2.weight',
            'self_attn.q_norm.weight': 'self_attention.q_layernorm.weight',
            'self_attn.k_norm.weight': 'self_attention.k_layernorm.weight',
        }
        
        for hf_subkey, megatron_subkey in simple_layer_map.items():
            if hf_subkey in layer_dict:
                hf_full_key = f'{hf_prefix}.{hf_subkey}'
                megatron_full_key = f'{prefix}.{megatron_subkey}'
                megatron_state_dict[megatron_full_key] = hf_state_dict[hf_full_key]
        
        # Handle MoE expert keys
        self._convert_experts(layer_idx, layer_dict, hf_state_dict, megatron_state_dict)
    
    def _convert_experts(self, layer_idx, layer_dict, hf_state_dict, megatron_state_dict):
        """Convert MoE expert weights."""
        prefix = f'decoder.layers.{layer_idx}'
        hf_prefix = f'model.layers.{layer_idx}'
        
        # Check for expert pattern
        expert_pattern = re.compile(r'mlp\.experts\.(\d+)\.(.+)')
        
        expert_weights = {}  # {expert_id: {subkey: tensor}}
        
        for subkey in layer_dict.keys():
            m = expert_pattern.match(subkey)
            if m:
                expert_id = int(m.group(1))
                weight_type = m.group(2)
                if expert_id not in expert_weights:
                    expert_weights[expert_id] = {}
                expert_weights[expert_id][weight_type] = hf_state_dict[f'{hf_prefix}.{subkey}']
        
        if not expert_weights:
            return
        
        num_experts = len(expert_weights)
        
        # Merge gate/up for each expert
        fc1_weights = []
        fc2_weights = []
        
        for expert_id in range(num_experts):
            if expert_id in expert_weights:
                exp = expert_weights[expert_id]
                if 'gate_proj.weight' in exp and 'up_proj.weight' in exp:
                    fc1 = merge_gate_up_proj(exp['gate_proj.weight'], exp['up_proj.weight'])
                    fc1_weights.append(fc1)
                if 'down_proj.weight' in exp:
                    fc2_weights.append(exp['down_proj.weight'])
        
        if fc1_weights:
            # Stack into [num_experts, hidden_size, ffn_size * 2]
            megatron_state_dict[f'{prefix}.mlp.experts.experts.linear_fc1.weight'] = torch.stack(fc1_weights, dim=0)
        if fc2_weights:
            # Stack into [num_experts, ffn_size, hidden_size]
            megatron_state_dict[f'{prefix}.mlp.experts.experts.linear_fc2.weight'] = torch.stack(fc2_weights, dim=0)
        
        # Handle router weights
        router_map = {
            'mlp.gate.weight': 'mlp.router.weight',
            'mlp.gate.e_score_correction_bias': 'mlp.router.expert_bias',
        }
        for hf_subkey, megatron_subkey in router_map.items():
            if hf_subkey in layer_dict:
                megatron_state_dict[f'{prefix}.{megatron_subkey}'] = hf_state_dict[f'{hf_prefix}.{hf_subkey}']
    
    def _save_dcp(self, state_dict):
        """Save state dict in torch_dist format (compatible with Megatron --ckpt-format torch_dist)."""
        try:
            # Initialize fake distributed environment for single-process saving
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend='gloo',
                    init_method='tcp://localhost:29500',
                    world_size=1,
                    rank=0
                )
            
            from megatron.core import dist_checkpointing
            
            os.makedirs(self.dst, exist_ok=True)
            
            # dist_checkpointing.save expects plain tensors for non-sharded single-rank save
            dist_checkpointing.save(
                sharded_state_dict=state_dict,
                checkpoint_dir=self.dst,
            )
            logging.info(f"Saved torch_dist checkpoint to {self.dst}")
        except Exception as e:
            # Fallback to simple torch checkpoint
            logging.warning(f"Could not save in torch_dist format: {e}") 
            logging.info("Falling back to torch.save format...")
            ckpt_path = os.path.join(self.dst, 'model_weights.pt')
            torch.save(state_dict, ckpt_path)
            logging.info(f"Saved model weights to {ckpt_path}")
    
    def _save_common(self):
        """Save common.pt with args."""
        common_path = os.path.join(self.dst, 'common.pt')
        torch.save({'args': self.args}, common_path)
        logging.info(f"Saved common.pt to {common_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace checkpoint to Megatron DCP format'
    )
    
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='Path to HuggingFace checkpoint directory',
    )
    parser.add_argument(
        '--target-path',
        type=str,
        required=True,
        help='Path to save Megatron DCP checkpoint',
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # All model args are read from HF config.json automatically
    converter = HFToDCPConverter(
        hf_ckpt_path=args.checkpoint_path,
        dst=args.target_path,
        megatron_args={},
    )
    
    converter.convert()
    logging.info("Conversion complete!")


if __name__ == '__main__':
    main()
