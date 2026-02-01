# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import atexit, json
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split

IGNORE_INDEX = -100


class SFTLowLevelDataset:
    """The low-level dataset loading jsonl data for SFT

    Args:
        dataset_path (str): The path to jsonl data
            Each line of the jsonl must have key "messages" (List[Dict]),
            which is a sequence of system/user/assistant messages.
            Must be in the following format:
            [
                {"role": "system", "content": "something"},
                {"role": "user", "content": "something1"},
                {"role": "assistant", "content": "something2"},
            ]
            A jsonl line can contain multiple conversations packed together into on list. Each
            conversation starts with the system role, and conversations can have multiple turns
            of the user and assistant roles.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "SFTDataset currently requires datasets library to be installed"
            )
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Return sample with messages and optional tools.
        
        Returns:
            dict: {"messages": [...], "tools": [...] or None}
        """
        item = self.dataset[idx]
        return {
            "messages": item["messages"],
            "tools": item.get("tools", None),
        }


class SFTDataset(MegatronDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return SFTLowLevelDataset(dataset_path)

    def __len__(self) -> int:
        return self.num_samples

    def _split_conversations(self, merged_conversations):
        split_conversations = []
        current = []
        for msg in merged_conversations:
            # Whenever we see a new system message, start a new conversation
            if msg["role"] == "system":
                if current:  # If previously accumulating a conversation, then store it
                    split_conversations.append(current)
                current = [msg]  # Then start the new conversation
            else:
                current.append(msg) # Continue accumulating the current conversation
        if current:  # Store any remaining conversation
            split_conversations.append(current)
        return split_conversations

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length

        # Get sample data (may contain messages and optional tools)
        sample_data = self.dataset[int(self.indices[idx % len(self.indices)])]
        
        # Handle both dict format {"messages": [...], "tools": [...]} 
        # and legacy list format (just messages)
        if isinstance(sample_data, dict):
            merged_conversations = sample_data["messages"]
            tools = sample_data.get("tools", None)
        else:
            # Legacy format: sample_data is just the messages list
            merged_conversations = sample_data
            tools = None
            
        split_conversations = self._split_conversations(merged_conversations)

        def extend_with_padding(tokens, targets, positions, pad_len):
            tokens.extend([pad] * pad_len)
            targets.extend([pad] * pad_len)
            positions.extend(range(positions[-1]+1, positions[-1]+1+pad_len))

        pack_tokens = []
        pack_targets = []
        pack_positions = []
        cu_seqlens = [0]
        eod = tokenizer.eod
        pad = tokenizer.pad
        # TODO(duncan): Track number of convs dropped and/or truncated and amount of end-padding
        for conversation in split_conversations:

            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True, add_generation_prompt=False, tools=tools
            )

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()

            # Add EOD, unless it's already present
            if tokens_list[-1] != eod:
                tokens_list.append(eod)
                targets_list.append(eod)

            pack_tokens.extend(tokens_list)
            pack_targets.extend(targets_list)

            assert not self.config.reset_position_ids
            pack_positions.extend(range(len(tokens_list)))

            if self.config.context_parallel_size > 1:
                pad_granularity = self.config.context_parallel_size * 2
                mod_token_count = len(pack_tokens) % pad_granularity
                if mod_token_count != 0:
                    pad_len = pad_granularity - mod_token_count
                    extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)

            # TODO(duncan): Consider also padding to multiple of number of tokens here. This might
            # be needed for efficiency (and potentially set via command-line argument).

            cu_seqlens.append(len(pack_tokens))

            # Handle any necessary truncation
            if len(pack_tokens) >= pack_length + 1:  # +1 here to account for later alignment
                truncate_left_not_right = True  # TODO(duncan): plumb this switch in
                if truncate_left_not_right:  # Retain existing eod
                    max_body = pack_length
                    pack_tokens = pack_tokens[-max_body:]
                    pack_targets = pack_targets[-max_body:]
                    pack_tokens.append(pad)
                    pack_targets.append(pad)
                else:  # Truncate right (need to add eod)
                    max_body = pack_length - 1
                    pack_tokens = pack_tokens[:max_body]
                    pack_targets = pack_targets[:max_body]
                    pack_tokens.extend([eod, pad])
                    pack_targets.extend([eod, pad])
                pack_positions = pack_positions[:pack_length+1]
                # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
                cu_seqlens[-1] = len(pack_tokens) - 1
                break

        # Handle any necessary padding
        if len(pack_tokens) < pack_length + 1:  # +1 here to account for later alignment
            pad_len = pack_length + 1 - len(pack_tokens)
            extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
            # Note len({pack_tokens, pack_targets, pack_positions}) should be pack_length + 1
            cu_seqlens[-1] = len(pack_tokens) - 1

        assert len(pack_tokens) == pack_length + 1
        assert len(pack_targets) == pack_length + 1
        assert len(pack_positions) == pack_length + 1

        # Align and convert to tensors
        input_ids    = torch.tensor(pack_tokens[:-1],  dtype=torch.int64)
        labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
        position_ids = torch.tensor(pack_positions[:-1], dtype=torch.int64)

        # Loss mask.
        loss_mask = torch.ones(pack_length, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0  # Mask paddings
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts

        # TODO(duncan): Optionally create an attention mask
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask
        # attention_mask = None

        assert len(cu_seqlens) >= 2
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
        # Calculating max_seqlen here, rather than incrementally above, because of possible
        # effects of truncation and padding
        adjacent_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = adjacent_diffs.max()  # max_seqlen is a 0-D tensor

        return {
            'tokens': input_ids,
            'labels': labels,
            # 'attention_mask': attention_mask,  # PyTorch collate cannot handle NoneType
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen,
        }


class NeatSFTDataset(SFTDataset):
    """SFT Dataset with neat packing support using first-fit decreasing algorithm.
    
    This dataset packs multiple shorter sequences into single bins to improve GPU
    utilization by reducing padding waste. All samples are pre-processed and packed
    during initialization.
    
    The max_sequences_per_pack value is read from global args (--sft-max-sequences-per-pack).
    
    Args:
        dataset: Low-level dataset
        dataset_path: Path to the dataset
        indices: Sample indices
        num_samples: Number of samples
        index_split: Train/valid/test split
        config: GPT dataset configuration
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        
        # Get max_sequences_per_pack from global args
        try:
            from megatron.training import get_args
            args = get_args()
            self.max_sequences_per_pack = getattr(args, 'sft_max_sequences_per_pack', None)
        except Exception:
            self.max_sequences_per_pack = None
        
        self._packed_samples = None
        self._packing_info = None
        self._dataset_path = dataset_path
        
        # Generate cache path
        self._cache_path = self._get_cache_path()
        
        # Try to load from cache, otherwise pack and cache
        if not self._load_from_cache():
            self._initialize_packing()
            self._save_to_cache()

    def _get_cache_path(self) -> Optional[str]:
        """Generate cache file path based on dataset configuration."""
        import hashlib
        import os
        
        if self._dataset_path is None:
            return None
        
        # Create a hash of the configuration that affects packing
        config_str = f"{self._dataset_path}:{len(self.indices)}:{self.config.sequence_length}:{self.max_sequences_per_pack}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Cache directory next to dataset
        cache_dir = os.path.join(os.path.dirname(self._dataset_path), ".sft_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Include rank in cache path for distributed training
        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0
        
        cache_file = f"packed_sft_{config_hash}_rank{rank}.pt"
        return os.path.join(cache_dir, cache_file)

    def _load_from_cache(self) -> bool:
        """Try to load packed samples from cache. Returns True if successful."""
        import os
        
        if self._cache_path is None or not os.path.exists(self._cache_path):
            return False
        
        try:
            cache_data = torch.load(self._cache_path, weights_only=False)
            self._packed_samples = cache_data['packed_samples']
            self._packing_info = cache_data.get('packing_info')
            print(f"[NeatSFTDataset] Loaded {len(self._packed_samples)} packed samples from cache: {self._cache_path}")
            return True
        except Exception as e:
            print(f"[NeatSFTDataset] Failed to load cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save packed samples to cache."""
        if self._cache_path is None or self._packed_samples is None:
            return
        
        try:
            cache_data = {
                'packed_samples': self._packed_samples,
                'packing_info': self._packing_info,
            }
            torch.save(cache_data, self._cache_path)
            print(f"[NeatSFTDataset] Saved {len(self._packed_samples)} packed samples to cache: {self._cache_path}")
        except Exception as e:
            print(f"[NeatSFTDataset] Failed to save cache: {e}")

    def _initialize_packing(self) -> None:
        """Pre-process and pack all samples."""
        from megatron.training.datasets.sft_packing import SFTNeatPacker
        
        # Tokenize all samples first
        all_samples = []
        for idx in range(len(self.indices)):
            sample = self._tokenize_single_sample(idx)
            if sample is not None:
                all_samples.append(sample)
        
        if not all_samples:
            self._packed_samples = []
            return
        
        # Create packer and pack samples
        tokenizer = self.config.tokenizer
        packer = SFTNeatPacker(
            bin_size=self.config.sequence_length,
            pad_token=tokenizer.pad,
            max_sequences_per_bin=self.max_sequences_per_pack,
        )
        
        self._packed_samples, self._packing_info = packer.pack_samples(all_samples)

    def _tokenize_single_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Tokenize a single sample without packing."""
        tokenizer = self.config.tokenizer
        pack_length = self.config.sequence_length
        
        merged_conversations = self.dataset[int(self.indices[idx % len(self.indices)])]
        split_conversations = self._split_conversations(merged_conversations)
        
        if not split_conversations:
            return None
        
        # Only use the first conversation for neat packing
        # (each sample = one conversation, packer combines multiple)
        conversation = split_conversations[0]
        
        tokens, targets = tokenizer.tokenize_conversation(
            conversation, return_target=True, add_generation_prompt=False
        )
        
        tokens_list = tokens.tolist()
        targets_list = targets.tolist()
        
        eod = tokenizer.eod
        pad = tokenizer.pad
        
        # Add EOD if not present
        if tokens_list[-1] != eod:
            tokens_list.append(eod)
            targets_list.append(eod)
        
        # Truncate if too long
        if len(tokens_list) > pack_length:
            tokens_list = tokens_list[:pack_length - 1] + [eod]
            targets_list = targets_list[:pack_length - 1] + [eod]
        
        # Create loss mask
        loss_mask = [1.0] * len(tokens_list)
        for i, t in enumerate(targets_list):
            if t == pad or t == IGNORE_INDEX:
                loss_mask[i] = 0.0
        
        return {
            'tokens': torch.tensor(tokens_list, dtype=torch.int64),
            'labels': torch.tensor(targets_list, dtype=torch.int64),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
        }

    def __len__(self) -> int:
        """Return number of packed bins."""
        if self._packed_samples is None:
            return 0
        return len(self._packed_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a packed sample by index."""
        if self._packed_samples is None or idx >= len(self._packed_samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        packed_sample = self._packed_samples[idx]
        
        # Shift labels for next-token prediction
        tokens = packed_sample['tokens']
        labels = packed_sample['labels']
        
        # For training, labels should be shifted by 1 relative to input
        # input_ids = tokens[:-1], labels = tokens[1:]
        input_ids = tokens.clone()
        labels_shifted = labels.clone()
        
        return {
            'tokens': input_ids,
            'labels': labels_shifted,
            'loss_mask': packed_sample['loss_mask'],
            'position_ids': packed_sample['position_ids'],
            'cu_seqlens': packed_sample['cu_seqlens'],
            'max_seqlen': packed_sample['max_seqlen'],
        }

