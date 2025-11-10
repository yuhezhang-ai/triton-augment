"""
Correctness tests comparing Triton-Augment with torchvision.transforms.v2.

These tests ensure that our implementation matches torchvision exactly.

Author: yuhezhang-ai
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# Skip all tests if CUDA or torchvision not available
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not TORCHVISION_AVAILABLE, reason="torchvision not available"),
]


class TestBrightnessCorrectness:
    """Test brightness adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("brightness_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_brightness_matches_torchvision(self, brightness_factor, shape):
        """Test that brightness adjustment matches torchvision exactly."""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply torchvision
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        
        # Apply triton-augment
        ta_result = F.adjust_brightness(img, brightness_factor)
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Brightness mismatch for factor={brightness_factor}, shape={shape}"
        )


class TestContrastCorrectness:
    """Test contrast adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_contrast_matches_torchvision(self, contrast_factor, shape):
        """Test that contrast adjustment matches torchvision exactly."""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply torchvision
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        
        # Apply triton-augment
        ta_result = F.adjust_contrast(img, contrast_factor)
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Contrast mismatch for factor={contrast_factor}, shape={shape}"
        )


class TestContrastFastCorrectness:
    """Test fast contrast adjustment (does NOT match torchvision)."""
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_contrast_fast_formula(self, contrast_factor, shape):
        """Test that fast contrast uses correct formula: (pixel - 0.5) * factor + 0.5"""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply triton fast contrast
        ta_result = F.adjust_contrast_fast(img, contrast_factor)
        
        # Manual calculation
        expected = (img - 0.5) * contrast_factor + 0.5
        expected = torch.clamp(expected, 0.0, 1.0)
        
        # Compare results (using PyTorch default tolerances)
        torch.testing.assert_close(
            ta_result,
            expected,
            msg=f"Fast contrast mismatch for factor={contrast_factor}, shape={shape}"
        )
    
    def test_contrast_fast_differs_from_torchvision(self):
        """Verify that fast contrast produces different results from torchvision."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        contrast_factor = 1.5
        
        # Apply both methods
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_fast_result = F.adjust_contrast_fast(img, contrast_factor)
        
        # They should NOT match (different algorithms)
        # We expect differences, so catch the assertion error
        try:
            torch.testing.assert_close(ta_fast_result, tv_result, rtol=1e-3, atol=1e-3)
            # If we get here, they matched unexpectedly
            assert False, "Fast contrast should differ from torchvision"
        except AssertionError as e:
            # Expected - they should be different
            if "should differ" in str(e):
                raise  # Re-raise our custom assertion
            # Otherwise it's the expected mismatch
            pass


class TestSaturationCorrectness:
    """Test saturation adjustment correctness against torchvision."""
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_saturation_matches_torchvision(self, saturation_factor, shape):
        """Test that saturation adjustment matches torchvision exactly."""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply torchvision
        tv_result = tvF.adjust_saturation(img, saturation_factor)
        
        # Apply triton-augment
        ta_result = F.adjust_saturation(img, saturation_factor)
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Saturation mismatch for factor={saturation_factor}, shape={shape}"
        )


class TestFusedCorrectness:
    """
    Test fused operations correctness.
    
    NOTE: Fused kernel uses FAST contrast (centered scaling), not torchvision's
    blend-with-mean. We test against sequential triton operations instead.
    """
    
    def test_fused_matches_sequential_triton(self):
        """Test that fused kernel matches sequential triton operations (with fast contrast)."""
        # Create test image
        img = torch.rand(4, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Parameters
        brightness_factor = 1.2
        contrast_factor = 1.1
        saturation_factor = 0.9
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Apply triton operations sequentially (using FAST contrast)
        ta_seq = F.adjust_brightness(img, brightness_factor)
        ta_seq = F.adjust_contrast_fast(ta_seq, contrast_factor)  # FAST contrast
        ta_seq = F.adjust_saturation(ta_seq, saturation_factor)
        ta_seq = F.normalize(ta_seq, mean=mean, std=std)
        
        # Apply triton-augment fused
        ta_fused = F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            mean=mean,
            std=std,
        )
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(
            ta_fused,
            ta_seq,
            msg="Fused kernel doesn't match sequential triton operations"
        )
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("size", [64, 128, 224])
    def test_fused_different_sizes(self, batch_size, size):
        """Test fused operation with different batch sizes and image sizes."""
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
        
        brightness_factor = 1.3
        contrast_factor = 0.9
        saturation_factor = 1.1
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        # Sequential triton (with fast contrast)
        ta_seq = F.adjust_brightness(img, brightness_factor)
        ta_seq = F.adjust_contrast_fast(ta_seq, contrast_factor)
        ta_seq = F.adjust_saturation(ta_seq, saturation_factor)
        ta_seq = F.normalize(ta_seq, mean=mean, std=std)
        
        # Triton-augment fused
        ta_fused = F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            mean=mean,
            std=std,
        )
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(ta_fused, ta_seq)
    
    def test_fused_without_contrast(self):
        """Test fused operation without contrast (should match torchvision exactly)."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        brightness_factor = 1.2
        saturation_factor = 0.9
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Torchvision (no contrast)
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        tv_result = tvF.adjust_saturation(tv_result, saturation_factor)
        mean_t = torch.tensor(mean, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        tv_result = (tv_result - mean_t) / std_t
        
        # Triton fused (no contrast)
        ta_result = F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Identity (no contrast)
            saturation_factor=saturation_factor,
            mean=mean,
            std=std,
        )
        
        # Should match torchvision exactly (no fast contrast involved)
        torch.testing.assert_close(ta_result, tv_result)


class TestEdgeCases:
    """Test edge cases and identity operations."""
    
    def test_identity_operations(self):
        """Test that identity parameters produce same results."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        # Brightness identity (factor=1.0)
        result = F.adjust_brightness(img, 1.0)
        torch.testing.assert_close(result, img, rtol=1e-5, atol=1e-5)
        
        # Contrast identity (factor=1.0)
        result = F.adjust_contrast(img, 1.0)
        torch.testing.assert_close(result, img, rtol=1e-4, atol=1e-4)
        
        # Saturation identity (factor=1.0)
        result = F.adjust_saturation(img, 1.0)
        torch.testing.assert_close(result, img, rtol=1e-4, atol=1e-4)
    
    def test_grayscale_saturation(self):
        """Test saturation=0 produces grayscale."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        # Apply saturation=0
        result = F.adjust_saturation(img, 0.0)
        
        # Check that all channels are equal (grayscale)
        r, g, b = result[:, 0], result[:, 1], result[:, 2]
        torch.testing.assert_close(r, g, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g, b, rtol=1e-4, atol=1e-4)
    
    def test_brightness_zero(self):
        """Test brightness=0 produces black image."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        # Apply brightness=0
        result = F.adjust_brightness(img, 0.0)
        
        # Check that result is all zeros
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=1e-5, atol=1e-5)


class TestTransformClasses:
    """Test transform classes match torchvision behavior."""
    
    def test_color_jitter_ranges(self):
        """Test that parameter ranges are computed correctly."""
        # Test symmetric range
        transform = ta.TritonColorJitter(brightness=0.2, contrast=0.3, saturation=0.1)
        assert transform.brightness == (0.8, 1.2), f"Expected (0.8, 1.2), got {transform.brightness}"
        assert transform.contrast == (0.7, 1.3), f"Expected (0.7, 1.3), got {transform.contrast}"
        assert transform.saturation == (0.9, 1.1), f"Expected (0.9, 1.1), got {transform.saturation}"
        
        # Test custom range
        transform = ta.TritonColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.2))
        assert transform.brightness == (0.5, 1.5)
        assert transform.contrast == (0.8, 1.2)
    
    def test_color_jitter_produces_valid_output(self):
        """Test that color jitter produces valid output shapes."""
        transform = ta.TritonColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.device == img.device
        assert result.dtype == img.dtype


class TestRGBToGrayscale:
    """Test RGB to grayscale conversion matches torchvision."""
    
    def test_grayscale_formula(self):
        """Test that our RGB to grayscale formula matches torchvision."""
        img = torch.rand(2, 3, 64, 64, device='cuda', dtype=torch.float32)
        
        # Torchvision grayscale
        tv_gray = tvF.rgb_to_grayscale(img, num_output_channels=1)
        
        # Our internal grayscale function
        from triton_augment.functional import _rgb_to_grayscale
        ta_gray = _rgb_to_grayscale(img)
        
        # Compare
        torch.testing.assert_close(ta_gray, tv_gray, rtol=1e-5, atol=1e-5)


class TestGrayscaleCorrectness:
    """Test grayscale conversion correctness against torchvision."""
    
    @pytest.mark.parametrize("num_output_channels", [1, 3])
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224)])
    def test_rgb_to_grayscale_matches_torchvision(self, num_output_channels, shape):
        """Test that rgb_to_grayscale matches torchvision exactly."""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply torchvision
        tv_result = tvF.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        
        # Apply triton-augment
        ta_result = F.rgb_to_grayscale(img, num_output_channels=num_output_channels)
        
        # Compare results
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Grayscale mismatch for num_output_channels={num_output_channels}, shape={shape}"
        )
        
        # Verify output shape
        expected_shape = (shape[0], num_output_channels, shape[2], shape[3])
        assert ta_result.shape == expected_shape
    
    def test_grayscale_deterministic(self):
        """Test that grayscale conversion is deterministic."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Apply twice
        result1 = F.rgb_to_grayscale(img, num_output_channels=3)
        result2 = F.rgb_to_grayscale(img, num_output_channels=3)
        
        # Should be identical
        torch.testing.assert_close(result1, result2)
    
    def test_random_grayscale_with_p_zero(self):
        """Test that random_grayscale with p=0 returns original image."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        result = F.random_grayscale(img, p=0.0)
        
        # Should be unchanged
        torch.testing.assert_close(result, img)
    
    def test_random_grayscale_with_p_one(self):
        """Test that random_grayscale with p=1 always converts to grayscale."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        result = F.random_grayscale(img, p=1.0, num_output_channels=3)
        
        # Should be grayscale (all channels identical)
        assert torch.allclose(result[:, 0], result[:, 1])
        assert torch.allclose(result[:, 1], result[:, 2])
    
    def test_fused_with_random_grayscale(self):
        """Test that fused_color_normalize with random_grayscale_p=1.0 produces grayscale."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Apply with random_grayscale_p=1.0 (always grayscale)
        result = F.fused_color_normalize(
            img,
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,  # This should be overridden
            random_grayscale_p=1.0,  # Force grayscale
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # After normalization, check if it was grayscale before normalize
        # Unnormalize to check
        mean_t = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
        unnormalized = result * std_t + mean_t
        
        # Should be grayscale (all channels should have similar values after accounting for different normalization)
        # Actually, after normalization with different mean/std per channel, they won't be identical
        # So let's test the saturation=0 effect directly
        
        # Better test: compare with explicit saturation=0
        result_explicit = F.fused_color_normalize(
            img,
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.0,  # Explicit grayscale
            random_grayscale_p=0.0,  # No random
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # The random_grayscale_p=1.0 should match saturation_factor=0.0
        torch.testing.assert_close(result, result_explicit, rtol=1e-5, atol=1e-5)


class TestGrayscaleTransforms:
    """Test grayscale transform classes."""
    
    def test_triton_grayscale_class(self):
        """Test TritonGrayscale transform class."""
        from triton_augment.transforms import TritonGrayscale
        
        transform = TritonGrayscale(num_output_channels=3)
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        result = transform(img)
        
        # Should be grayscale
        assert result.shape == (2, 3, 128, 128)
        assert torch.allclose(result[:, 0], result[:, 1])
        assert torch.allclose(result[:, 1], result[:, 2])
    
    def test_triton_random_grayscale_class(self):
        """Test TritonRandomGrayscale transform class."""
        from triton_augment.transforms import TritonRandomGrayscale
        
        # Test with p=0
        transform_never = TritonRandomGrayscale(p=0.0, num_output_channels=3)
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        result = transform_never(img)
        torch.testing.assert_close(result, img)
        
        # Test with p=1
        transform_always = TritonRandomGrayscale(p=1.0, num_output_channels=3)
        result = transform_always(img)
        assert torch.allclose(result[:, 0], result[:, 1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

