import argparse
import gc
import json
import logging
import multiprocessing
import os
import re
import shutil

import torch
from safetensors.torch import save_file as safe_save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def set_up_planner(self, state_dict, metadata=None, is_coordinator=False) -> None:
    assert not state_dict
    assert metadata is not None

    # rebuild the state dict from the metadata
    for k, v in metadata.state_dict_metadata.items():
        if k not in self.keys:
            continue

        if isinstance(v, TensorStorageMetadata):
            v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
        state_dict[k] = v

    super(
        torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner, self
    ).set_up_planner(state_dict, metadata, is_coordinator)


torch.distributed.checkpoint.default_planner._EmptyStateDictLoadPlanner.set_up_planner = set_up_planner


def per_block_cast_to_fp8(x: torch.Tensor, scale_ue8m0):
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_scale = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    if scale_ue8m0:
        x_scale = (
            x_scale.maximum(torch.tensor(1e-10, device=x.device)).log2().ceil().exp2()
        )
    x_scaled = (x_view / x_scale).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), x_scale.view(
        x_view.size(0), x_view.size(2)
    )


def transform_qkv_weight(args, weight):
    each_kv_size = args.kv_channels
    each_q_size = args.kv_channels * args.num_attention_heads // args.num_query_groups

    qs = []
    ks = []
    vs = []

    for qkv in torch.chunk(weight, args.num_query_groups, dim=0):
        q, g, k, v = qkv.split(
            [each_q_size, each_q_size, each_kv_size, each_kv_size], dim=0
        )

        q_split = q.split(args.kv_channels, dim=0)
        g_split = g.split(args.kv_channels, dim=0)

        for _q, _g in zip(q_split, g_split):
            qs.append(_q)
            qs.append(_g)

        ks.append(k)
        vs.append(v)

    all_q = torch.cat(qs, dim=0)
    all_k = torch.cat(ks, dim=0)
    all_v = torch.cat(vs, dim=0)

    return all_q, all_k, all_v


def parse_args():
    parser = argparse.ArgumentParser(
        description='config for convert dcp checkpoint script'
    )

    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        required=True,
        help='Path tp DCP checkpoint.',
    )
    parser.add_argument(
        '--target-path',
        type=str,
        default=None,
        required=True,
        help='Path to save checkpoint',
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        required=True,
        help='Path which contains tokenizer_config.json and other tokenizer files',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path which contains config.json and modeling/configuration scripts',
    )
    parser.add_argument(
        '--quant-mlp',
        action='store_true',
        default=False,
        help='Whether to quantize MLP weights',
    )
    parser.add_argument(
        '--quant-attn',
        action='store_true',
        default=False,
        help='Whether to quantize self attention weights. It involves --quant-mlp.',
    )
    return parser.parse_args()


class DCPKeyMapper:
    layer_pattern = re.compile(r'decoder\.layers\.(\d+)\.(.+)')
    hf_layer_prefix = 'model.layers'
    key_map = {
        'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
        'decoder.final_layernorm.weight': 'model.norm.weight',
        'output_layer.weight': 'lm_head.weight',
        # map for layers, split layer idx
        'self_attention.linear_qkv.layer_norm_weight': 'input_layernorm.weight',
        'mlp.linear_fc1.layer_norm_weight': 'post_attention_layernorm.weight',
        'pre_mlp_layernorm.weight': 'post_attention_layernorm.weight',
        'mlp.router.expert_bias': 'mlp.gate.e_score_correction_bias',
        'mlp.router.weight': 'mlp.gate.weight',
        'self_attention.q_layernorm.weight': 'self_attn.q_norm.weight',
        'self_attention.k_layernorm.weight': 'self_attn.k_norm.weight',
        'self_attention.linear_proj.weight': 'self_attn.o_proj.weight',
        'mlp.linear_fc2.weight': 'mlp.down_proj.weight',
        # for mtp
        'enorm.weight': 'enorm.weight',
        'hnorm.weight': 'hnorm.weight',
        'eh_proj.weight': 'eh_proj.weight',
        'final_layernorm.weight': 'final_layernorm.weight',
    }
    split_key_map = {
        'mlp.linear_fc1.weight': 'mlp',
    }
    qkv_key_map = {
        'self_attention.linear_qkv.weight': (
            'self_attn.q_proj.weight',
            'self_attn.k_proj.weight',
            'self_attn.v_proj.weight',
        ),
    }
    expert_key_map = {
        'mlp.experts.experts.linear_fc1.weight': 'mlp.experts.{expert_id}',
        'mlp.experts.experts.linear_fc2.weight': 'mlp.experts.{expert_id}.down_proj.weight',
    }
    fp8_weight = set()

    mtp_pattern = re.compile(r'mtp\.layers\.(\d+)\.(transformer_layer\.)?(.+)')

    @classmethod
    def set_fp8_weight(cls, args):
        if args.quant_mlp:
            cls.fp8_weight.add('mlp.shared_experts.up_proj.weight')
            cls.fp8_weight.add('mlp.shared_experts.gate_proj.weight')
            cls.fp8_weight.add('mlp.shared_experts.down_proj.weight')
            cls.fp8_weight.add('mlp.up_proj.weight')
            cls.fp8_weight.add('mlp.down_proj.weight')
            cls.fp8_weight.add('mlp.gate_proj.weight')

            for i in range(args.num_experts):
                cls.fp8_weight.add(f'mlp.experts.{i}.gate_proj.weight')
                cls.fp8_weight.add(f'mlp.experts.{i}.up_proj.weight')
                cls.fp8_weight.add(f'mlp.experts.{i}.down_proj.weight')

            if args.quant_attn:
                cls.fp8_weight.add('self_attn.q_proj.weight')
                cls.fp8_weight.add('self_attn.k_proj.weight')
                cls.fp8_weight.add('self_attn.v_proj.weight')
                cls.fp8_weight.add('self_attn.o_proj.weight')

    @classmethod
    def split_gate_up_proj(cls, key, value):
        if value is None:
            return [f'{key}.gate_proj.weight', f'{key}.up_proj.weight']
        else:
            gate_proj, up_proj = torch.chunk(value, 2, dim=0)
            return {
                f'{key}.gate_proj.weight': gate_proj,
                f'{key}.up_proj.weight': up_proj,
            }

    @classmethod
    def parse_key_or_kv(cls, key, value):
        if value is None:
            return [key]
        else:
            return {key: value}

    @classmethod
    def transform_qkv(cls, key, value, args):
        if value is None:
            return [*key]
        else:
            return {k: v for k, v in zip(key, transform_qkv_weight(args, value))}

    @classmethod
    def merge_list_or_dict(cls, arr):
        if isinstance(arr[0], list):
            return [x for a in arr for x in a]
        else:
            return {k: v for a in arr for k, v in a.items()}

    @classmethod
    def map_kv(self, key, value, args):
        m = re.match(self.layer_pattern, key)
        if m:
            # weight in layers
            layer_idx, post_key = m.groups()
            layer_prefix = f'{self.hf_layer_prefix}.{layer_idx}'
            if post_key in self.split_key_map:
                updated = self.split_gate_up_proj(self.split_key_map[post_key], value)
            elif post_key in self.qkv_key_map:
                updated = self.transform_qkv(self.qkv_key_map[post_key], value, args)
            elif post_key in self.expert_key_map:

                def split_expert(ei):
                    expert_value = None if value is None else value[ei]
                    return self.split_gate_up_proj(
                        f'{self.expert_key_map[post_key].format(expert_id=ei)}',
                        expert_value,
                    )

                if 'down_proj' in self.expert_key_map[post_key]:
                    updated = self.merge_list_or_dict(
                        [
                            self.parse_key_or_kv(
                                f'{self.expert_key_map[post_key].format(expert_id=ei)}',
                                None if value is None else value[ei],
                            )
                            for ei in range(args.num_experts)
                        ]
                    )
                else:
                    updated = self.merge_list_or_dict(
                        [split_expert(ei) for ei in range(args.num_experts)]
                    )
            else:
                updated = self.parse_key_or_kv(f'{self.key_map[post_key]}', value)
            if value is None:
                return [
                    f'{layer_prefix}.{x}'
                    for k in updated
                    for x in self.quant_if_needed(k, None, args)
                ]
            else:
                return {
                    f'{layer_prefix}.{new_k}': new_v
                    for k, v in updated.items()
                    for new_k, new_v in self.quant_if_needed(k, v, args).items()
                }
        elif m := re.match(self.mtp_pattern, key):
            return self.map_kv_mtp(key, value, args)

        # weight out of layers
        if key in (
            'embedding.word_embeddings.weight',
            'output_layer.weight',
        ) and isinstance(value, torch.Tensor):
            value = value[: args.vocab_size, :]

        return self.parse_key_or_kv(self.key_map[key], value)

    @classmethod
    def quant_if_needed(cls, key, value, args):
        if value is None:
            if key in cls.fp8_weight:
                return [key, f'{key}_scale_inv']
            else:
                return [key]

        if key in cls.fp8_weight:
            scale_ue8m0 = False
            if key == 'self_attn.o_proj.weight':
                scale_ue8m0 = True
            if 'mlp.experts' in key and 'down_proj' not in key:
                scale_ue8m0 = True
            qw, scale = per_block_cast_to_fp8(value, scale_ue8m0)
            return {key: qw, f'{key}_scale_inv': scale}

        else:
            return {key: value}

    @classmethod
    def map_kv_mtp(cls, key, value, args):
        m = re.match(cls.mtp_pattern, key)
        if m:
            layer_idx, is_transformer, subkey = m.groups()

            assert layer_idx == '0', (
                f'Only 1-layer mtp is supported，current {key=} current {layer_idx=}'
            )
            layer_idx = args.num_layers

            if subkey in cls.split_key_map:
                updated = cls.split_gate_up_proj(cls.split_key_map[subkey], value)
            elif subkey in cls.qkv_key_map:
                updated = cls.transform_qkv(cls.qkv_key_map[subkey], value, args)
            elif subkey in cls.expert_key_map:

                def split_expert(ei):
                    expert_value = None if value is None else value[ei]
                    return cls.split_gate_up_proj(
                        cls.expert_key_map[subkey].format(expert_id=ei), expert_value
                    )

                if 'down_proj' in cls.expert_key_map[subkey]:
                    updated = cls.merge_list_or_dict(
                        [
                            cls.parse_key_or_kv(
                                f'{cls.expert_key_map[subkey].format(expert_id=ei)}',
                                None if value is None else value[ei],
                            )
                            for ei in range(args.num_experts)
                        ]
                    )
                else:
                    updated = cls.merge_list_or_dict(
                        [split_expert(ei) for ei in range(args.num_experts)]
                    )
            else:
                updated = cls.parse_key_or_kv(cls.key_map[subkey], value)
            if value is None:
                return [
                    f'{cls.hf_layer_prefix}.{layer_idx}.{x}'
                    for k in updated
                    for x in cls.quant_if_needed(k, None, args)
                ]
            else:
                return {
                    f'{cls.hf_layer_prefix}.{layer_idx}.{new_k}': new_v
                    for k, v in updated.items()
                    for new_k, new_v in cls.quant_if_needed(k, v, args).items()
                }
        else:
            raise Exception(f'Unsupported key {key}')


class SafetensorPart:
    def __init__(self, idx, total, keys, nbytes, converter):
        self.part_filename = f'model-{idx + 1:05d}-of-{total:05d}.safetensors'
        self.ckpt = converter.ckpt
        self.keys = keys
        self.nbytes = nbytes
        self.dst = converter.dst
        self.args = converter.args
        self.idx = idx

    def rewrite(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        torch.set_num_threads(1)
        DCPKeyMapper.set_fp8_weight(self.args)
        logging.info(f'convert for {self.part_filename} with {self.nbytes} bytes')
        state_dict = _load_state_dict_from_keys(self.keys, checkpoint_id=self.ckpt)

        logging.info('load done')

        new_state_dict = {
            new_k: new_v
            for k, v in state_dict.items()
            for new_k, new_v in DCPKeyMapper.map_kv(k, v, self.args).items()
        }

        logging.info('convert done')
        safe_save_file(
            new_state_dict,
            os.path.join(self.dst, self.part_filename),
            metadata={'format': 'pt'},
        )
        logging.info('save done')
        index_map = {k: self.part_filename for k in new_state_dict}
        del state_dict
        del new_state_dict
        gc.collect()
        return index_map

    def gen_index_map(self):
        new_keys = [
            new_k
            for k in self.keys
            for new_k in DCPKeyMapper.map_kv(k, None, self.args)
        ]
        return {k: self.part_filename for k in new_keys}


def rewrite_part(part):
    return part.rewrite()


class DCPConverter:
    def __init__(
        self,
        ckpt,
        dst,
        quant_mlp: bool,
        quant_attn: bool,
        vocab_size,
    ):
        self.ckpt = ckpt
        self.dst = dst
        if not os.path.exists(dst):
            os.mkdir(dst)

        self.args = torch.load(os.path.join(ckpt, 'common.pt'), weights_only=False)[
            'args'
        ]

        self.args.quant_mlp = quant_mlp
        self.args.quant_attn = quant_attn
        self.args.vocab_size = vocab_size

        if not hasattr(self.args, 'first_k_dense_replace'):
            self.args.first_k_dense_replace = self.args.moe_layer_freq.index(1)

        self.reader = FileSystemReader(self.ckpt)

    def key_to_keep(self, key):
        return (
            'optimizer' not in key
            and '_extra_state' not in key
            and 'rng_state' not in key
            and 'chained_0' not in key
            and 'rerun_state_machine_state' not in key
        )

    def part_generator(self, threshold=10 * 2**30):
        meta = [
            (k, v.properties.dtype.itemsize * v.size.numel())
            for k, v in self.reader.read_metadata().state_dict_metadata.items()
            if self.key_to_keep(k)
        ]
        start = 0
        total_bytes = 0
        for i, (k, size_in_bytes) in enumerate(meta):
            if total_bytes + size_in_bytes > threshold:
                yield {x[0] for x in meta[start:i]}, total_bytes
                total_bytes = size_in_bytes
                start = i
            else:
                total_bytes += size_in_bytes
        if start < len(meta):
            yield {k for k, _ in meta[start:]}, sum([v for _, v in meta[start:]])

    def rewrite_to_safetensors(self):
        logging.info('start rewrite safetensors...')
        parts = [(i, k, v) for i, (k, v) in enumerate(self.part_generator())]
        total_bytes = sum([nbytes for _, _, nbytes in parts])
        meta = {'metadata': {'total_size': total_bytes}, 'weight_map': {}}
        nparts = len(parts)
        parts = [SafetensorPart(i, nparts, k, v, self) for i, k, v in parts]

        if len(parts) > 0:
            multiprocessing.set_start_method('spawn')
            with multiprocessing.Pool(64) as pool:
                pool.map(rewrite_part, parts)

        meta['weight_map'] = {
            k: v for part in parts for k, v in part.gen_index_map().items()
        }
        with open(os.path.join(self.dst, 'model.safetensors.index.json'), 'w') as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    args = parse_args()

    if args.quant_attn:
        args.quant_mlp = True

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    torch.set_num_threads(1)

    vocab_size = json.load(
        open(os.path.join(args.tokenizer_path, 'tokenizer_config.json'))
    )['vocab_size']

    converter = DCPConverter(
        args.checkpoint_path,
        args.target_path,
        args.quant_mlp,
        args.quant_attn,
        vocab_size,
    )
    DCPKeyMapper.set_fp8_weight(converter.args)
    converter.rewrite_to_safetensors()

    tokenizer_files = [
        'generation_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'additional_chat_templates',
        'chat_template.jinja',
    ]

    model_files = [
        'configuration_gauss3_moe.py',
        'modeling_gauss3_moe.py',
    ]

    def copy_file(path, file):
        full_path = os.path.join(path, file)
        target_path = os.path.join(args.target_path, file)

        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                shutil.copytree(full_path, target_path)

            else:
                shutil.copy(full_path, target_path)

        else:
            logging.info(f'copy: {file} does not exist.')

    for f in tokenizer_files:
        copy_file(args.tokenizer_path, f)

    for f in model_files:
        copy_file(args.model_path, f)

    config = json.load(open(os.path.join(args.model_path, 'config.json')))

    config['num_nextn_predict_layers'] = 1

    if args.quant_mlp:
        quant_config = {
            'activation_scheme': 'dynamic',
            'fmt': 'e4m3',
            'quant_method': 'fp8',
            'weight_block_size': [128, 128],
        }

        if not args.quant_attn:
            quant_config['ignored_layers'] = ['model.layers.*.self_attn']

        config['quantization_config'] = quant_config

    json.dump(
        config,
        open(os.path.join(args.target_path, 'config.json'), 'w'),
        ensure_ascii=False,
        indent=2,
    )