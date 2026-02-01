# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Unit tests for SFT Neat Packing."""

import pytest
import torch

from megatron.training.datasets.sft_packing import SFTNeatPacker


class TestSFTNeatPacker:
    """Tests for the SFTNeatPacker class."""

    def test_basic_packing(self):
        """Test basic packing of samples into bins."""
        packer = SFTNeatPacker(
            bin_size=100,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        # Create samples with varying lengths
        samples = [
            {
                'tokens': torch.arange(1, 31, dtype=torch.int64),  # length 30
                'labels': torch.arange(1, 31, dtype=torch.int64),
                'loss_mask': torch.ones(30, dtype=torch.float32),
            },
            {
                'tokens': torch.arange(1, 41, dtype=torch.int64),  # length 40
                'labels': torch.arange(1, 41, dtype=torch.int64),
                'loss_mask': torch.ones(40, dtype=torch.float32),
            },
            {
                'tokens': torch.arange(1, 21, dtype=torch.int64),  # length 20
                'labels': torch.arange(1, 21, dtype=torch.int64),
                'loss_mask': torch.ones(20, dtype=torch.float32),
            },
        ]

        packed_samples, packing_info = packer.pack_samples(samples)

        # All 3 samples should fit in one bin (30 + 40 + 20 = 90 < 100)
        assert len(packed_samples) == 1
        assert packed_samples[0]['tokens'].shape[0] == 100  # padded to bin_size
        
        # Check packing info
        assert packing_info.num_original_samples == 3
        assert packing_info.num_packed_bins == 1

    def test_multiple_bins(self):
        """Test that samples are split across multiple bins when necessary."""
        packer = SFTNeatPacker(
            bin_size=50,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        # Create samples that won't fit in one bin
        samples = [
            {
                'tokens': torch.arange(1, 31, dtype=torch.int64),  # length 30
                'labels': torch.arange(1, 31, dtype=torch.int64),
                'loss_mask': torch.ones(30, dtype=torch.float32),
            },
            {
                'tokens': torch.arange(1, 31, dtype=torch.int64),  # length 30
                'labels': torch.arange(1, 31, dtype=torch.int64),
                'loss_mask': torch.ones(30, dtype=torch.float32),
            },
        ]

        packed_samples, packing_info = packer.pack_samples(samples)

        # 30 + 30 = 60 > 50, so need 2 bins
        assert len(packed_samples) == 2
        assert packing_info.num_packed_bins == 2

    def test_max_sequences_per_bin(self):
        """Test that max_sequences_per_bin limit is respected."""
        packer = SFTNeatPacker(
            bin_size=100,
            pad_token=0,
            max_sequences_per_bin=2,
        )

        # Create 4 small samples
        samples = [
            {
                'tokens': torch.arange(1, 11, dtype=torch.int64),  # length 10
                'labels': torch.arange(1, 11, dtype=torch.int64),
                'loss_mask': torch.ones(10, dtype=torch.float32),
            }
            for _ in range(4)
        ]

        packed_samples, packing_info = packer.pack_samples(samples)

        # With max 2 sequences per bin, need 2 bins for 4 samples
        assert len(packed_samples) == 2
        
        # Each bin should have 2 sequences (check cu_seqlens)
        for packed in packed_samples:
            # cu_seqlens has len = num_sequences + 1
            assert len(packed['cu_seqlens']) == 3  # 2 sequences + 1

    def test_cu_seqlens_correctness(self):
        """Test that cu_seqlens are correctly computed."""
        packer = SFTNeatPacker(
            bin_size=100,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        samples = [
            {
                'tokens': torch.arange(1, 21, dtype=torch.int64),  # length 20
                'labels': torch.arange(1, 21, dtype=torch.int64),
                'loss_mask': torch.ones(20, dtype=torch.float32),
            },
            {
                'tokens': torch.arange(1, 31, dtype=torch.int64),  # length 30
                'labels': torch.arange(1, 31, dtype=torch.int64),
                'loss_mask': torch.ones(30, dtype=torch.float32),
            },
        ]

        packed_samples, _ = packer.pack_samples(samples)

        # Both samples fit in one bin
        assert len(packed_samples) == 1
        
        cu_seqlens = packed_samples[0]['cu_seqlens'].tolist()
        # First-fit decreasing: longer sample (30) comes first, then shorter (20)
        # So cu_seqlens should be [0, 30, 50]
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 30
        assert cu_seqlens[2] == 50

    def test_position_ids(self):
        """Test that position_ids are correctly reset for each sequence."""
        packer = SFTNeatPacker(
            bin_size=100,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        samples = [
            {
                'tokens': torch.arange(1, 21, dtype=torch.int64),
                'labels': torch.arange(1, 21, dtype=torch.int64),
                'loss_mask': torch.ones(20, dtype=torch.float32),
            },
            {
                'tokens': torch.arange(1, 16, dtype=torch.int64),
                'labels': torch.arange(1, 16, dtype=torch.int64),
                'loss_mask': torch.ones(15, dtype=torch.float32),
            },
        ]

        packed_samples, _ = packer.pack_samples(samples)
        position_ids = packed_samples[0]['position_ids']

        # First-fit decreasing: 20-token sample first (positions 0-19), 
        # then 15-token sample (positions 0-14)
        # Check first sequence positions
        assert position_ids[0].item() == 0
        assert position_ids[19].item() == 19
        
        # Check second sequence positions (reset to 0)
        assert position_ids[20].item() == 0
        assert position_ids[34].item() == 14

    def test_empty_samples(self):
        """Test handling of empty sample list."""
        packer = SFTNeatPacker(
            bin_size=100,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        packed_samples, packing_info = packer.pack_samples([])

        assert len(packed_samples) == 0
        assert packing_info.num_original_samples == 0
        assert packing_info.num_packed_bins == 0

    def test_truncation_of_long_sequence(self):
        """Test that sequences longer than bin_size are truncated."""
        packer = SFTNeatPacker(
            bin_size=50,
            pad_token=0,
            max_sequences_per_bin=4,
        )

        samples = [
            {
                'tokens': torch.arange(1, 101, dtype=torch.int64),  # length 100 > bin_size
                'labels': torch.arange(1, 101, dtype=torch.int64),
                'loss_mask': torch.ones(100, dtype=torch.float32),
            },
        ]

        packed_samples, packing_info = packer.pack_samples(samples)

        # Should have 1 bin with truncated sequence
        assert len(packed_samples) == 1
        assert packed_samples[0]['tokens'].shape[0] == 50
        assert packing_info.num_truncated_samples == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
