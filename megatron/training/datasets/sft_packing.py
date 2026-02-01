# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SFT Neat Packing utilities.

This module implements first-fit decreasing bin packing algorithm for SFT datasets
to improve GPU utilization by minimizing padding waste.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from megatron.core.packed_seq_params import PackedSeqParams


logger = logging.getLogger(__name__)


@dataclass
class SFTPackingInfo:
    """Information about how sequences are packed into bins.
    
    Attributes:
        bin_seq_indices: List where each element contains the sequence indices in that bin
        seq_starts: Dict mapping bin index to list of start positions for each sequence
        seq_lengths: List of all original sequence lengths
        num_bins: Total number of bins created
    """
    bin_seq_indices: List[List[int]]
    seq_starts: Dict[int, List[int]]
    seq_lengths: List[int]
    num_bins: int


class SFTNeatPacker:
    """Packs multiple SFT sequences into bins using first-fit decreasing algorithm.
    
    This packer sorts sequences by length in descending order and greedily assigns
    each sequence to the first bin that has enough remaining capacity.
    
    Args:
        bin_size: Maximum number of tokens per bin (typically sequence_length)
        pad_token: Token ID used for padding
        max_sequences_per_bin: Maximum number of sequences allowed in a single bin
    """

    def __init__(
        self,
        bin_size: int,
        pad_token: int,
        max_sequences_per_bin: int = 8,
    ):
        self.bin_size = bin_size
        self.pad_token = pad_token
        self.max_sequences_per_bin = max_sequences_per_bin

    def pack_samples(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Tuple[List[Dict[str, torch.Tensor]], SFTPackingInfo]:
        """Pack multiple SFT samples into bins using first-fit decreasing.
        
        Args:
            samples: List of tokenized samples, each containing:
                - 'tokens': Token IDs tensor
                - 'labels': Label tensor
                - 'loss_mask': Loss mask tensor
                
        Returns:
            Tuple of (packed_samples, packing_info):
                - packed_samples: List of packed samples with cu_seqlens and max_seqlen
                - packing_info: SFTPackingInfo with packing metadata
        """
        if not samples:
            return [], SFTPackingInfo([], {}, [], 0)

        # Get sequence lengths (excluding padding)
        seq_lengths = []
        for sample in samples:
            tokens = sample['tokens']
            # Find actual length by looking for padding
            if hasattr(tokens, 'tolist'):
                tokens_list = tokens.tolist() if tokens.dim() == 1 else tokens.squeeze().tolist()
            else:
                tokens_list = list(tokens)
            
            # Count non-padding tokens
            length = len(tokens_list)
            for i in range(len(tokens_list) - 1, -1, -1):
                if tokens_list[i] != self.pad_token:
                    length = i + 1
                    break
            seq_lengths.append(length)

        # Sort indices by length in descending order (first-fit decreasing)
        sorted_indices = sorted(
            range(len(samples)), 
            key=lambda i: seq_lengths[i], 
            reverse=True
        )

        # Initialize bins
        bins: List[List[int]] = []  # Each bin contains list of sample indices
        bin_remaining: List[int] = []  # Remaining capacity in each bin

        # Assign samples to bins using first-fit decreasing
        for idx in sorted_indices:
            seq_len = seq_lengths[idx]
            
            if seq_len > self.bin_size:
                # Sequence too long, will be truncated later
                logger.warning(
                    f"Sequence {idx} has length {seq_len} > bin_size {self.bin_size}, "
                    "will be truncated"
                )
                seq_len = self.bin_size

            # Find first bin that fits
            placed = False
            for bin_idx in range(len(bins)):
                if (bin_remaining[bin_idx] >= seq_len and 
                    len(bins[bin_idx]) < self.max_sequences_per_bin):
                    bins[bin_idx].append(idx)
                    bin_remaining[bin_idx] -= seq_len
                    placed = True
                    break
            
            if not placed:
                # Create new bin
                bins.append([idx])
                bin_remaining.append(self.bin_size - seq_len)

        # Create packed samples
        packed_samples = []
        seq_starts_dict: Dict[int, List[int]] = {}

        for bin_idx, bin_indices in enumerate(bins):
            packed_tokens = []
            packed_labels = []
            packed_loss_mask = []
            packed_positions = []
            seq_starts = []
            current_pos = 0

            for sample_idx in bin_indices:
                sample = samples[sample_idx]
                seq_len = seq_lengths[sample_idx]
                
                # Truncate to actual length
                tokens = sample['tokens'][:seq_len]
                labels = sample['labels'][:seq_len]
                loss_mask = sample['loss_mask'][:seq_len]

                seq_starts.append(current_pos)
                
                # Convert to lists for extending
                if hasattr(tokens, 'tolist'):
                    packed_tokens.extend(tokens.tolist())
                    packed_labels.extend(labels.tolist())
                    packed_loss_mask.extend(loss_mask.tolist())
                else:
                    packed_tokens.extend(list(tokens))
                    packed_labels.extend(list(labels))
                    packed_loss_mask.extend(list(loss_mask))
                
                # Position IDs reset for each sequence
                packed_positions.extend(range(seq_len))
                current_pos += seq_len

            seq_starts.append(current_pos)  # End position
            seq_starts_dict[bin_idx] = seq_starts

            # Pad to bin_size
            pad_len = self.bin_size - len(packed_tokens)
            if pad_len > 0:
                packed_tokens.extend([self.pad_token] * pad_len)
                packed_labels.extend([self.pad_token] * pad_len)
                packed_loss_mask.extend([0.0] * pad_len)
                # Continue position IDs for padding
                if packed_positions:
                    last_pos = packed_positions[-1]
                    packed_positions.extend(range(last_pos + 1, last_pos + 1 + pad_len))
                else:
                    packed_positions.extend(range(pad_len))

            # Build cu_seqlens for this bin
            cu_seqlens = [0]
            for i, sample_idx in enumerate(bin_indices):
                cu_seqlens.append(seq_starts[i + 1])
            
            # Max sequence length in this bin
            max_seqlen = max(
                seq_lengths[idx] for idx in bin_indices
            ) if bin_indices else 0

            packed_sample = {
                'tokens': torch.tensor(packed_tokens[:self.bin_size], dtype=torch.int64),
                'labels': torch.tensor(packed_labels[:self.bin_size], dtype=torch.int64),
                'loss_mask': torch.tensor(packed_loss_mask[:self.bin_size], dtype=torch.float32),
                'position_ids': torch.tensor(packed_positions[:self.bin_size], dtype=torch.int64),
                'cu_seqlens': torch.tensor(cu_seqlens, dtype=torch.int32),
                'max_seqlen': torch.tensor(max_seqlen, dtype=torch.int32),
            }
            packed_samples.append(packed_sample)

        # Create packing info
        packing_info = SFTPackingInfo(
            bin_seq_indices=bins,
            seq_starts=seq_starts_dict,
            seq_lengths=seq_lengths,
            num_bins=len(bins),
        )

        # Log packing statistics
        self._log_packing_stats(samples, packed_samples, packing_info, seq_lengths)

        return packed_samples, packing_info

    def _log_packing_stats(
        self,
        original_samples: List[Dict],
        packed_samples: List[Dict],
        packing_info: SFTPackingInfo,
        seq_lengths: List[int],
    ) -> None:
        """Log packing efficiency statistics."""
        if not packed_samples:
            return

        total_tokens = sum(seq_lengths)
        total_capacity = len(packed_samples) * self.bin_size
        packing_efficiency = total_tokens / total_capacity if total_capacity > 0 else 0
        
        seqs_per_bin = [len(indices) for indices in packing_info.bin_seq_indices]
        avg_seqs_per_bin = sum(seqs_per_bin) / len(seqs_per_bin) if seqs_per_bin else 0
        
        logger.info(f"[SFT Neat Packing] Statistics:")
        logger.info(f"  - Original samples: {len(original_samples)}")
        logger.info(f"  - Packed bins: {len(packed_samples)}")
        logger.info(f"  - Bin size: {self.bin_size} tokens")
        logger.info(f"  - Average sequences per bin: {avg_seqs_per_bin:.1f}")
        logger.info(f"  - Min/Max sequences per bin: {min(seqs_per_bin)}/{max(seqs_per_bin)}")
        logger.info(
            f"  - Packing efficiency: {packing_efficiency:.1%} "
            f"({total_tokens:,} / {total_capacity:,} tokens)"
        )
        logger.info(
            f"  - Compression ratio: {len(original_samples) / len(packed_samples):.2f}x"
        )


def create_packed_seq_params_for_sft(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    bin_size: int,
    device: torch.device,
) -> PackedSeqParams:
    """Create PackedSeqParams for SFT packed sequence.
    
    Args:
        cu_seqlens: Cumulative sequence lengths tensor
        max_seqlen: Maximum sequence length in the bin
        bin_size: Size of the bin
        device: Device to create tensors on
        
    Returns:
        PackedSeqParams for Transformer Engine attention
    """
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
    
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
