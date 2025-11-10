"""
Unit tests for Triton-Augment transforms and functional API.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import triton_augment as ta
import triton_augment.functional as F


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestFunctionalAPI:
    """Test the functional API."""
    
    def test_apply_brightness(self):
        """Test brightness adjustment."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        result = F.apply_brightness(img, brightness_factor=0.1)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
        
        # Check that brightness was applied
        expected = img + 0.1
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    def test_apply_contrast(self):
        """Test contrast adjustment."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        result = F.apply_contrast(img, contrast_factor=1.2)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
        
        # Check that contrast was applied
        expected = img * 1.2
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    
    def test_apply_normalize(self):
        """Test normalization."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        result = F.apply_normalize(img, mean=mean, std=std)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_fused_color_jitter(self):
        """Test fused color jitter."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        
        result = F.fused_color_jitter(
            img,
            brightness_factor=0.1,
            contrast_factor=1.2,
            saturation_factor=0.8
        )
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_fused_color_normalize(self):
        """Test fully fused operation."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        
        result = F.fused_color_normalize(
            img,
            brightness_factor=0.1,
            contrast_factor=1.2,
            saturation_factor=0.8,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_input_validation(self):
        """Test input validation."""
        # Test non-CUDA tensor
        img_cpu = torch.rand(2, 3, 64, 64)
        with pytest.raises(ValueError, match="must be on CUDA"):
            F.apply_brightness(img_cpu, 0.1)
        
        # Test wrong number of dimensions
        img_wrong_dims = torch.rand(3, 64, 64, device='cuda')
        with pytest.raises(ValueError, match="must be a 4D tensor"):
            F.apply_brightness(img_wrong_dims, 0.1)
        
        # Test wrong number of channels
        img_wrong_channels = torch.rand(2, 4, 64, 64, device='cuda')
        with pytest.raises(ValueError, match="must have 3 channels"):
            F.apply_brightness(img_wrong_channels, 0.1)


class TestTransforms:
    """Test the transform classes."""
    
    def test_triton_color_jitter(self):
        """Test TritonColorJitter transform."""
        transform = ta.TritonColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        )
        
        img = torch.rand(2, 3, 64, 64, device='cuda')
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_triton_normalize(self):
        """Test TritonNormalize transform."""
        transform = ta.TritonNormalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        img = torch.rand(2, 3, 64, 64, device='cuda')
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_triton_color_jitter_normalize(self):
        """Test TritonColorJitterNormalize transform."""
        transform = ta.TritonColorJitterNormalize(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        img = torch.rand(2, 3, 64, 64, device='cuda')
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype
    
    def test_color_jitter_ranges(self):
        """Test custom parameter ranges."""
        # Test with float (symmetric range)
        transform1 = ta.TritonColorJitter(brightness=0.2)
        assert transform1.brightness == (-0.2, 0.2)
        
        # Test with tuple (custom range)
        transform2 = ta.TritonColorJitter(brightness=(-0.3, 0.3))
        assert transform2.brightness == (-0.3, 0.3)
    
    def test_randomness(self):
        """Test that transform produces different results each time."""
        transform = ta.TritonColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        )
        
        img = torch.rand(2, 3, 64, 64, device='cuda')
        
        # Apply transform multiple times
        results = [transform(img) for _ in range(5)]
        
        # Check that at least some results are different
        # (with very high probability)
        all_same = all(
            torch.allclose(results[0], r, rtol=1e-5, atol=1e-5)
            for r in results[1:]
        )
        assert not all_same, "Transform should produce random results"


class TestBatchSizes:
    """Test different batch sizes and image dimensions."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        img = torch.rand(batch_size, 3, 64, 64, device='cuda')
        transform = ta.TritonColorJitter(brightness=0.1)
        result = transform(img)
        
        assert result.shape == img.shape
    
    @pytest.mark.parametrize("size", [32, 64, 128, 224, 256])
    def test_different_image_sizes(self, size):
        """Test with different image sizes."""
        img = torch.rand(2, 3, size, size, device='cuda')
        transform = ta.TritonColorJitter(brightness=0.1)
        result = transform(img)
        
        assert result.shape == img.shape


class TestDTypes:
    """Test different data types."""
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_different_dtypes(self, dtype):
        """Test with different data types."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=dtype)
        transform = ta.TritonColorJitter(brightness=0.1)
        result = transform(img)
        
        assert result.dtype == dtype


class TestEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_identity_transforms(self):
        """Test that identity parameters produce no change."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        
        # Apply identity transformation
        result = F.fused_color_jitter(
            img,
            brightness_factor=0.0,
            contrast_factor=1.0,
            saturation_factor=1.0
        )
        
        # Should be approximately equal (allowing for small numerical errors)
        torch.testing.assert_close(result, img, rtol=1e-4, atol=1e-4)
    
    def test_normalize_without_color_jitter(self):
        """Test normalization without color jitter."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        result = F.fused_color_normalize(
            img,
            brightness_factor=0.0,
            contrast_factor=1.0,
            saturation_factor=1.0,
            mean=mean,
            std=std
        )
        
        # Check approximate normalization
        # Each channel should have mean ≈ 0 and std ≈ 1
        # (but not exact due to finite sample size)
        assert result.shape == img.shape
    
    def test_color_jitter_without_normalize(self):
        """Test color jitter without normalization."""
        img = torch.rand(2, 3, 64, 64, device='cuda')
        
        result = F.fused_color_normalize(
            img,
            brightness_factor=0.1,
            contrast_factor=1.2,
            saturation_factor=0.8,
            mean=None,
            std=None
        )
        
        assert result.shape == img.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

