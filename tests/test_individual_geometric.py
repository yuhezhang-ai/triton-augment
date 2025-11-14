"""
Tests for individual geometric operations (functional + transform classes).

Compares Triton-Augment with torchvision to ensure correctness.
Tests both functional API (F.crop, F.horizontal_flip) and transform classes.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestCropCorrectness:
    """Test crop operations match torchvision exactly."""
    
    @pytest.mark.parametrize("top,left,height,width", [
        (0, 0, 100, 100),           # Top-left corner
        (10, 20, 80, 90),            # Offset crop
        (50, 50, 50, 50),            # Center-ish crop
        (0, 0, 224, 224),            # Full image (identity)
        (100, 150, 24, 24),          # Small crop
    ])
    def test_crop_functional_matches_torchvision(self, top, left, height, width):
        """Test that F.crop matches torchvision exactly."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        tv_result = tvF.crop(img, top, left, height, width)
        ta_result = F.crop(img, top, left, height, width)
        
        torch.testing.assert_close(ta_result, tv_result)
    
    @pytest.mark.parametrize("size", [112, (112, 112), (100, 150)])
    def test_center_crop_functional_matches_torchvision(self, size):
        """Test that F.center_crop matches torchvision exactly."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        tv_result = tvF.center_crop(img, size)
        ta_result = F.center_crop(img, size)
        
        torch.testing.assert_close(ta_result, tv_result)
    
    def test_crop_deterministic(self):
        """Test that crop is deterministic."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        result1 = F.crop(img, 50, 60, 100, 120)
        result2 = F.crop(img, 50, 60, 100, 120)
        
        torch.testing.assert_close(result1, result2)
    
    def test_triton_random_crop_produces_valid_output(self):
        """Test that TritonRandomCrop produces valid crops."""
        transform = ta.TritonRandomCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        assert result.dtype == img.dtype
        assert result.device == img.device
    
    def test_triton_center_crop_produces_valid_output(self):
        """Test that TritonCenterCrop produces valid crops."""
        transform = ta.TritonCenterCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        
        # Verify it's centered
        tv_result = tvF.center_crop(img, 112)
        torch.testing.assert_close(result, tv_result)


class TestFlipCorrectness:
    """Test flip operations match torchvision exactly."""
    
    def test_horizontal_flip_functional_matches_torchvision(self):
        """Test that F.horizontal_flip matches torchvision exactly."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        tv_result = tvF.horizontal_flip(img)
        ta_result = F.horizontal_flip(img)
        
        torch.testing.assert_close(ta_result, tv_result)
    
    def test_horizontal_flip_deterministic(self):
        """Test that horizontal_flip is deterministic."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        result1 = F.horizontal_flip(img)
        result2 = F.horizontal_flip(img)
        
        torch.testing.assert_close(result1, result2)
    
    def test_horizontal_flip_twice_is_identity(self):
        """Test that flipping twice returns the original image."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        flipped = F.horizontal_flip(img)
        restored = F.horizontal_flip(flipped)
        
        torch.testing.assert_close(restored, img)
    
    def test_triton_random_horizontal_flip_produces_valid_output(self):
        """Test that TritonRandomHorizontalFlip produces valid output."""
        transform = ta.TritonRandomHorizontalFlip(p=1.0)  # Always flip
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == img.shape
        
        # Verify it's flipped
        expected = F.horizontal_flip(img)
        torch.testing.assert_close(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

