"""
Tests for fused geometric operations (crop + flip).

Tests using fused_augment with no-op color parameters and
TritonRandomCropFlip transform class.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestFusedCropFlip:
    """Test fused crop+flip operations."""
    
    @pytest.mark.parametrize("flip", [False, True])
    @pytest.mark.parametrize("crop_params", [
        (20, 30, 100, 120),
        (50, 50, 112, 112),
        (0, 0, 200, 200),
    ])
    def test_fused_crop_flip_matches_sequential_triton(self, crop_params, flip):
        """
        Test that fused crop+flip matches sequential triton operations.
        
        Uses fused_augment with no-op color parameters.
        Validates fusion correctness for geometric operations.
        """
        img = torch.rand(4, 3, 224, 224, device='cuda')
        top, left, height, width = crop_params
        
        # Sequential: crop then flip
        seq_result = F.crop(img, top, left, height, width)
        if flip:
            seq_result = F.horizontal_flip(seq_result)
        
        # Fused
        fused_result = F.fused_augment(
            img, top, left, height, width, flip_horizontal=flip,
            brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0,  # No-op color
            grayscale=False, mean=None, std=None  # No-op
        )
        
        torch.testing.assert_close(fused_result, seq_result)
    
    def test_fused_crop_flip_matches_torchvision_sequential(self):
        """
        Test that fused crop+flip matches sequential torchvision operations.
        
        Validates correctness against torchvision baseline.
        """
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Sequential torchvision
        tv_result = tvF.crop(img, 20, 30, 100, 120)
        tv_result = tvF.horizontal_flip(tv_result)
        
        # Triton fused
        ta_result = F.fused_augment(
            img, 20, 30, 100, 120, flip_horizontal=True,
            brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0,  # No-op color
            grayscale=False, mean=None, std=None  # No-op
        )
        
        torch.testing.assert_close(ta_result, tv_result)
    
    @pytest.mark.parametrize("shape", [
        (2, 3, 224, 224),
        (1, 3, 512, 512),
        (8, 3, 128, 128),
    ])
    def test_fused_crop_flip_works_across_different_shapes(self, shape):
        """
        Test fused crop+flip with different tensor shapes.
        
        Validates robustness across various batch sizes and image dimensions.
        """
        img = torch.rand(*shape, device='cuda')
        _, _, h, w = shape
        
        crop_h, crop_w = h // 2, w // 2
        top, left = h // 4, w // 4
        
        # Sequential
        seq_result = F.crop(img, top, left, crop_h, crop_w)
        seq_result = F.horizontal_flip(seq_result)
        
        # Fused
        fused_result = F.fused_augment(
            img, top, left, crop_h, crop_w, flip_horizontal=True,
            brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0,  # No-op color
            grayscale=False, mean=None, std=None  # No-op
        )
        
        torch.testing.assert_close(fused_result, seq_result)

    def test_fused_augment_optional_crop(self):
        """
        Test fused augment with optional crop.
        """
        transform = ta.TritonFusedAugment(
            crop_size=None,
            horizontal_flip_p=1,
        )
        
        img = torch.rand(4, 3, 224, 224, device='cuda')
        out = transform(img)

        seq_result = F.horizontal_flip(img)

        torch.testing.assert_close(out, seq_result)

class TestRandomCropFlipTransform:
    """Test TritonRandomCropFlip transform class."""
    
    def test_random_crop_flip_produces_valid_output(self):
        """Test that TritonRandomCropFlip produces valid output."""
        transform = ta.TritonRandomCropFlip(112, horizontal_flip_p=1.0)  # Always flip
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        assert result.dtype == img.dtype
        assert result.device == img.device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

