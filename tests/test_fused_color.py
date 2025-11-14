"""
Tests for fused color operations (color jitter + normalize).

Tests using fused_augment with no-op geometric parameters and
TritonColorJitterNormalize transform class.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestFusedColorNormalize:
    """
    Test fused color+normalize operations.
    
    NOTE: Fused kernel uses FAST contrast (centered scaling), not torchvision's
    blend-with-mean.
    """
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("size", [64, 128, 224])
    def test_fused_color_matches_sequential_triton(self, batch_size, size):
        """
        Test that fused color+normalize matches sequential triton operations.
        
        Uses fused_augment with no-op geometric parameters (full image crop).
        Validates fusion correctness with FAST contrast mode.
        Parameterized across multiple batch sizes and image sizes.
        """
        img = torch.rand(batch_size, 3, size, size, device='cuda', dtype=torch.float32)
        
        brightness_factor = 1.3
        contrast_factor = 0.9
        saturation_factor = 1.1
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Sequential triton (with fast contrast)
        ta_seq = F.adjust_brightness(img, brightness_factor)
        ta_seq = F.adjust_contrast_fast(ta_seq, contrast_factor)
        ta_seq = F.adjust_saturation(ta_seq, saturation_factor)
        ta_seq = F.normalize(ta_seq, mean=mean, std=std)
        
        # Triton-augment fused (ultimate kernel with no-op geometric ops)
        _, _, height, width = img.shape
        ta_fused = F.fused_augment(
            img,
            top=0, left=0, height=height, width=width, flip_horizontal=False,  # No-op geometric
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            saturation_factor=saturation_factor,
            grayscale=False,
            mean=mean,
            std=std,
        )
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(ta_fused, ta_seq)
    
    @pytest.mark.parametrize("batch_size,height,width", [
        # Regular sizes
        (2, 128, 128),
        (1, 224, 224),
        # Odd sizes (not multiple of 2)
        (2, 223, 223),
        # Non-square
        (1, 64, 128),
        (2, 224, 112),
        (1, 197, 211),  # Both odd and non-square
        # Very small
        (1, 7, 7),
        (2, 13, 13),
        (1, 1, 1),  # Extreme: single pixel
        # Very large
        (1, 512, 512),
        (1, 1024, 768),
        (1, 4096, 4096),
        # Prime numbers (often edge cases)
        (1, 97, 97),
        (2, 101, 103),
    ])
    def test_fused_color_matches_torchvision_without_contrast(self, batch_size, height, width):
        """
        Test fused color+normalize matches torchvision on various irregular sizes.
        
        Skips contrast to ensure exact match (no FAST contrast difference).
        Tests robustness across unusual dimensions (odd, non-square, prime, etc.).
        """
        img = torch.rand(batch_size, 3, height, width, device='cuda', dtype=torch.float32)
        
        brightness_factor = 1.2
        saturation_factor = 0.9
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Torchvision (no contrast)
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        tv_result = tvF.adjust_saturation(tv_result, saturation_factor)
        tv_result = tvF.normalize(tv_result, mean=list(mean), std=list(std))
        
        # Triton fused (no contrast)
        _, _, h, w = img.shape
        ta_result = F.fused_augment(
            img,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Identity (no contrast)
            saturation_factor=saturation_factor,
            grayscale=False,
            mean=mean,
            std=std,
        )
        
        # Should match torchvision exactly (no fast contrast involved)
        torch.testing.assert_close(
            ta_result, 
            tv_result,
            msg=f"Mismatch for shape ({batch_size}, 3, {height}, {width})"
        )
    
    @pytest.mark.parametrize("saturation_factor", [0.5, 1.0, 1.2, 1.5])
    def test_fused_color_with_grayscale_matches_torchvision(self, saturation_factor):
        """
        Test fused color+normalize with grayscale matches torchvision sequential ops.
        
        Validates grayscale ordering: saturation is applied BEFORE grayscale conversion.
        """
        img = torch.rand(1, 3, 224, 224, device='cuda', dtype=torch.float32)
        
        brightness_factor = 1.2
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Torchvision (no contrast)
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        tv_result = tvF.adjust_saturation(tv_result, saturation_factor)
        tv_result = tvF.rgb_to_grayscale(tv_result, num_output_channels=3)
        tv_result = tvF.normalize(tv_result, mean=list(mean), std=list(std))
        
        # Triton fused (no contrast)
        _, _, h, w = img.shape
        ta_result = F.fused_augment(
            img,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Identity (no contrast)
            saturation_factor=saturation_factor,
            grayscale=True,
            mean=mean,
            std=std,
        )

        # Print the max difference - minor difference expected
        # Reason: in our fused kernel, we directly set saturation_factor to 0 when apply grayscale after saturation, but grayscale formula does not have all coefficients sum to 1 (0.2989 + 0.587 + 0.114 = 0.9999, not 1.0), this will cause a small difference in the result compared to sequential operations.
        print(f"Max difference: {torch.max(torch.abs(ta_result - tv_result))}")
        
        # Should match torchvision exactly (no fast contrast involved)
        torch.testing.assert_close(
            ta_result, 
            tv_result,
            msg=f"Mismatch for shape (1, 3, 224, 224) with saturation_factor={saturation_factor}"
        )
    
    def test_grayscale_ordering_saturation_then_grayscale(self):
        """
        Test that grayscale is applied AFTER saturation, not as a replacement.
        
        Critical correctness test: saturation can oversaturate/clamp values,
        and grayscale should operate on those clamped values, not raw RGB.
        """
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Test 1: saturation=2.0 (oversaturate) then grayscale
        _, _, h, w = img.shape
        result_sat_then_gray = F.fused_augment(
            img,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            saturation_factor=2.0,  # Oversaturate and clamp
            grayscale=True,  # Then convert to grayscale
        )
        
        # Test 2: grayscale directly (no saturation)
        result_gray_only = F.fused_augment(
            img,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric
            saturation_factor=1.0,  # No saturation
            grayscale=True,  # Just grayscale
        )
        
        # They should be DIFFERENT because saturation oversaturates/clamps first
        max_diff = torch.abs(result_sat_then_gray - result_gray_only).max().item()
        assert max_diff > 0.001, (
            f"Expected difference between 'saturation→grayscale' vs 'grayscale alone', "
            f"but max_diff={max_diff:.6f} is too small"
        )
        
        # Verify grayscale conversion worked (all channels equal)
        r, g, b = result_sat_then_gray[:, 0], result_sat_then_gray[:, 1], result_sat_then_gray[:, 2]
        torch.testing.assert_close(r, g, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(g, b, rtol=1e-6, atol=1e-6)


class TestColorJitterNormalizeTransform:
    """Test TritonColorJitterNormalize transform class."""
    
    def test_color_jitter_normalize_produces_valid_output(self):
        """Test that TritonColorJitterNormalize produces valid output."""
        transform = ta.TritonColorJitterNormalize(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        img = torch.rand(4, 3, 224, 224, device='cuda', dtype=torch.float32)
        result = transform(img)
        
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert result.device == img.device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

