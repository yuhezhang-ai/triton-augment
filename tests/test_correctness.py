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
        # Use structured input (not random) to ensure consistent difference
        img = torch.linspace(0, 1, 2*3*128*128, device='cuda').reshape(2, 3, 128, 128)
        contrast_factor = 1.5
        
        # Apply both methods
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_fast_result = F.adjust_contrast_fast(img, contrast_factor)
        
        # They should NOT be identical (different formulas)
        # Check that there's a meaningful difference
        max_diff = torch.abs(ta_fast_result - tv_result).max().item()
        
        # For structured data with factor=1.5, we expect meaningful difference
        assert max_diff > 0.01, f"Fast contrast too similar to torchvision (max_diff={max_diff})"


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


class TestNormalizeCorrectness:
    """Test normalization correctness against torchvision."""
    
    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 224, 224), (4, 3, 128, 128)])
    @pytest.mark.parametrize("mean,std", [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Simple
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # Identity
    ])
    def test_normalize_matches_torchvision(self, shape, mean, std):
        """Test that normalization matches torchvision exactly."""
        # Create test image
        img = torch.rand(*shape, device='cuda', dtype=torch.float32)
        
        # Apply torchvision normalize
        tv_result = tvF.normalize(img, mean=list(mean), std=list(std))
        
        # Apply triton-augment normalize
        ta_result = F.normalize(img, mean=mean, std=std)
        
        # Compare results (using PyTorch default tolerances: rtol=1e-05, atol=1e-08)
        torch.testing.assert_close(
            ta_result,
            tv_result,
            msg=f"Normalize mismatch for mean={mean}, std={std}, shape={shape}"
        )
    
    def test_normalize_deterministic(self):
        """Test that normalize produces consistent results."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        result1 = F.normalize(img, mean=mean, std=std)
        result2 = F.normalize(img, mean=mean, std=std)
        
        torch.testing.assert_close(result1, result2, msg="Normalize should be deterministic")


class TestFusedCorrectness:
    """
    Test fused operations correctness.
    
    NOTE: Fused kernel uses FAST contrast (centered scaling), not torchvision's
    blend-with-mean.
    """
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("size", [64, 128, 224])
    def test_fused_matches_sequential_triton(self, batch_size, size):
        """
        Test that fused kernel matches sequential triton operations (with fast contrast).
        
        This is the main correctness test for the fused kernel, parameterized across
        multiple batch sizes and image sizes.
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
    def test_fused_matches_torchvision_without_contrast(self, batch_size, height, width):
        """Test fused operation without contrast on various irregular sizes (should match torchvision exactly)."""
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
        ta_result = F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Identity (no contrast)
            saturation_factor=saturation_factor,
            mean=mean,
            std=std,
        )
        
        # Should match torchvision exactly (no fast contrast involved)
        torch.testing.assert_close(
            ta_result, 
            tv_result,
            msg=f"Mismatch for shape ({batch_size}, 3, {height}, {width})"
        )


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

    @pytest.mark.parametrize("saturation_factor", [0.5, 1.0, 1.2, 1.5])
    def test_fused_matches_torchvision_with_grayscale(self, saturation_factor):
        """Test fused operation without contrast on various irregular sizes (should match torchvision exactly)."""
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
        ta_result = F.fused_color_normalize(
            img,
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Identity (no contrast)
            saturation_factor=saturation_factor,
            random_grayscale_p=1.0,
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
    
    def test_ultimate_fused_with_grayscale_matches_torchvision(self):
        """Test that ultimate_fused_augment with random_grayscale matches torchvision sequential ops."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        
        # Crop parameters
        top, left = 20, 30
        height, width = 112, 112
        
        # Color parameters
        brightness_factor = 1.4
        saturation_factor = 0.6
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Apply ultimate fused with random_grayscale_p=1.0 (always grayscale)
        # Note: Uses FAST contrast (centered scaling), not torchvision contrast
        ta_result = F.ultimate_fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=True,
            brightness_factor=brightness_factor,
            contrast_factor=1.0,  # Skip contrast since we use FAST version
            saturation_factor=saturation_factor,
            random_grayscale_p=1.0,  # Force grayscale
            mean=mean,
            std=std
        )
        
        # Apply torchvision sequential operations
        # 1. Crop
        tv_result = tvF.crop(img, top, left, height, width)
        # 2. Horizontal flip
        tv_result = tvF.horizontal_flip(tv_result)
        # 3. Brightness
        tv_result = tvF.adjust_brightness(tv_result, brightness_factor)
        # 4. Saturation
        tv_result = tvF.adjust_saturation(tv_result, saturation_factor)
        # 5. Grayscale (convert to grayscale with 3 output channels)
        tv_result = tvF.rgb_to_grayscale(tv_result, num_output_channels=3)
        # 6. Normalize
        tv_result = tvF.normalize(tv_result, mean=list(mean), std=list(std))
        
        # Should match within float32 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-5, atol=1e-5)
    
    def test_fused_with_random_grayscale_ordering(self):
        """Test that random grayscale is applied AFTER saturation, not as replacement."""
        img = torch.rand(2, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Test 1: saturation=2.0 (oversaturate) then grayscale
        result_sat_then_gray = F.fused_color_normalize(
            img,
            saturation_factor=2.0,  # Oversaturate and clamp
            random_grayscale_p=1.0,  # Then convert to grayscale
        )
        
        # Test 2: grayscale directly (no saturation)
        result_gray_only = F.fused_color_normalize(
            img,
            saturation_factor=1.0,  # No saturation
            random_grayscale_p=1.0,  # Just grayscale
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


class TestFloat16Support:
    """Test float16 (half precision) correctness and comparison with float32."""
    
    @pytest.mark.parametrize("brightness_factor", [0.5, 1.0, 1.5])
    def test_brightness_float16_vs_float32(self, brightness_factor):
        """Test that float16 brightness matches float32 within appropriate tolerance."""
        # Create test images in both dtypes
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply brightness in both dtypes
        result_fp32 = F.adjust_brightness(img_fp32, brightness_factor)
        result_fp16 = F.adjust_brightness(img_fp16, brightness_factor)
        
        # Compare (float16 has lower precision, so use relaxed tolerance)
        # float16 has ~3-4 decimal digits of precision
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,  # 0.1% relative tolerance
            atol=1e-3   # 0.001 absolute tolerance
        )
    
    @pytest.mark.parametrize("contrast_factor", [0.5, 1.0, 1.5])
    def test_contrast_float16_vs_float32(self, contrast_factor):
        """Test that float16 contrast matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply contrast
        result_fp32 = F.adjust_contrast(img_fp32, contrast_factor)
        result_fp16 = F.adjust_contrast(img_fp16, contrast_factor)
        
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,
            atol=1e-3
        )
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5])
    def test_saturation_float16_vs_float32(self, saturation_factor):
        """Test that float16 saturation matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply saturation
        result_fp32 = F.adjust_saturation(img_fp32, saturation_factor)
        result_fp16 = F.adjust_saturation(img_fp16, saturation_factor)
        
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=1e-3,
            atol=1e-3
        )
    
    def test_normalize_float16_vs_float32(self):
        """Test that float16 normalization matches float32 within appropriate tolerance."""
        img_fp32 = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Apply normalization
        result_fp32 = F.normalize(img_fp32, mean, std)
        result_fp16 = F.normalize(img_fp16, mean, std)
        
        # Normalize involves division by small std values (~0.2), which amplifies
        # float16 precision errors. Relaxed tolerance for precision comparison.
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=0.01,   # 1% relative tolerance (relaxed from 0.1%)
            atol=0.01    # 0.01 absolute tolerance (relaxed from 0.001)
        )
    
    def test_fused_float16_vs_float32(self):
        """Test that fused kernel produces similar results for float16 vs float32."""
        img_fp32 = torch.rand(4, 3, 224, 224, device='cuda', dtype=torch.float32)
        img_fp16 = img_fp32.to(torch.float16)
        
        # Apply fused transform (uses FAST contrast)
        result_fp32 = F.fused_color_normalize(
            img_fp32,
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,
            random_grayscale_p=0.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        result_fp16 = F.fused_color_normalize(
            img_fp16,
            brightness_factor=1.2,
            contrast_factor=1.1,
            saturation_factor=0.9,
            random_grayscale_p=0.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        
        # Fused operation with normalize has same float16 precision issues
        # due to division by small std values
        torch.testing.assert_close(
            result_fp16.float(), 
            result_fp32, 
            rtol=0.01,   # 1% relative tolerance (relaxed from 0.1%)
            atol=0.01    # 0.01 absolute tolerance (relaxed from 0.001)
        )
    
    @pytest.mark.parametrize("brightness_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_brightness(self, brightness_factor):
        """Test that float16 brightness matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_brightness(img, brightness_factor)
        ta_result = F.adjust_brightness(img, brightness_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("saturation_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_saturation(self, saturation_factor):
        """Test that float16 saturation matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_saturation(img, saturation_factor)
        ta_result = F.adjust_saturation(img, saturation_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("contrast_factor", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_float16_matches_torchvision_contrast(self, contrast_factor):
        """Test that float16 contrast matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.adjust_contrast(img, contrast_factor)
        ta_result = F.adjust_contrast(img, contrast_factor)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("mean,std", [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),              # Simple
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),              # Identity
    ])
    def test_float16_matches_torchvision_normalize(self, mean, std):
        """Test that float16 normalize matches torchvision float16."""
        img = torch.rand(2, 3, 224, 224, device='cuda', dtype=torch.float16)
        
        # Apply both
        tv_result = tvF.normalize(img, mean=list(mean), std=list(std))
        ta_result = F.normalize(img, mean=mean, std=std)
        
        # Should match within float16 precision
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-3, atol=1e-3)


# ============================================================================
# Geometric Operations Correctness Tests
# ============================================================================


class TestCropCorrectness:
    """Test crop operations match torchvision exactly."""
    
    @pytest.mark.parametrize("top,left,height,width", [
        (0, 0, 100, 100),           # Top-left corner
        (10, 20, 80, 90),            # Offset crop
        (50, 50, 50, 50),            # Center-ish crop
        (0, 0, 224, 224),            # Full image (identity)
        (100, 150, 24, 24),          # Small crop
    ])
    def test_crop_matches_torchvision(self, top, left, height, width):
        """Test that crop matches torchvision exactly."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        tv_result = tvF.crop(img, top, left, height, width)
        ta_result = F.crop(img, top, left, height, width)
        
        torch.testing.assert_close(ta_result, tv_result)
    
    @pytest.mark.parametrize("size", [112, (112, 112), (100, 150)])
    def test_center_crop_matches_torchvision(self, size):
        """Test that center_crop matches torchvision exactly."""
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


class TestFlipCorrectness:
    """Test flip operations match torchvision exactly."""
    
    def test_horizontal_flip_matches_torchvision(self):
        """Test that horizontal_flip matches torchvision exactly."""
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


class TestFusedCropFlipCorrectness:
    """Test fused crop+flip operations."""
    
    @pytest.mark.parametrize("flip", [False, True])
    @pytest.mark.parametrize("crop_params", [
        (20, 30, 100, 120),
        (50, 50, 112, 112),
        (0, 0, 200, 200),
    ])
    def test_fused_crop_flip_matches_sequential_triton(self, crop_params, flip):
        """Test that fused_crop_flip matches sequential triton operations."""
        img = torch.rand(4, 3, 224, 224, device='cuda')
        top, left, height, width = crop_params
        
        # Sequential: crop then flip
        seq_result = F.crop(img, top, left, height, width)
        if flip:
            seq_result = F.horizontal_flip(seq_result)
        
        # Fused
        fused_result = F.fused_crop_flip(img, top, left, height, width, flip_horizontal=flip)
        
        torch.testing.assert_close(fused_result, seq_result)
    
    def test_fused_crop_flip_matches_sequential_torchvision(self):
        """Test that fused_crop_flip matches sequential torchvision operations."""
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Sequential torchvision
        tv_result = tvF.crop(img, 20, 30, 100, 120)
        tv_result = tvF.horizontal_flip(tv_result)
        
        # Triton fused
        ta_result = F.fused_crop_flip(img, 20, 30, 100, 120, flip_horizontal=True)
        
        torch.testing.assert_close(ta_result, tv_result)
    
    @pytest.mark.parametrize("shape", [
        (2, 3, 224, 224),
        (1, 3, 512, 512),
        (8, 3, 128, 128),
    ])
    def test_fused_crop_flip_different_shapes(self, shape):
        """Test fused_crop_flip with different tensor shapes."""
        img = torch.rand(*shape, device='cuda')
        _, _, h, w = shape
        
        crop_h, crop_w = h // 2, w // 2
        top, left = h // 4, w // 4
        
        # Sequential
        seq_result = F.crop(img, top, left, crop_h, crop_w)
        seq_result = F.horizontal_flip(seq_result)
        
        # Fused
        fused_result = F.fused_crop_flip(img, top, left, crop_h, crop_w, flip_horizontal=True)
        
        torch.testing.assert_close(fused_result, seq_result)


class TestGeometricTransformClasses:
    """Test geometric transform classes."""
    
    def test_random_crop_produces_valid_output(self):
        """Test that TritonRandomCrop produces valid crops."""
        from triton_augment.transforms import TritonRandomCrop
        
        transform = TritonRandomCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        assert result.dtype == img.dtype
        assert result.device == img.device
    
    def test_center_crop_produces_valid_output(self):
        """Test that TritonCenterCrop produces valid crops."""
        from triton_augment.transforms import TritonCenterCrop
        
        transform = TritonCenterCrop(112)
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        
        # Verify it's centered
        tv_result = tvF.center_crop(img, 112)
        torch.testing.assert_close(result, tv_result)
    
    def test_random_horizontal_flip_produces_valid_output(self):
        """Test that TritonRandomHorizontalFlip produces valid output."""
        from triton_augment.transforms import TritonRandomHorizontalFlip
        
        transform = TritonRandomHorizontalFlip(p=1.0)  # Always flip
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == img.shape
        
        # Verify it's flipped
        expected = F.horizontal_flip(img)
        torch.testing.assert_close(result, expected)
    
    def test_random_crop_flip_produces_valid_output(self):
        """Test that TritonRandomCropFlip produces valid output."""
        from triton_augment.transforms import TritonRandomCropFlip
        
        transform = TritonRandomCropFlip(112, horizontal_flip_p=1.0)  # Always flip
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        assert result.dtype == img.dtype
        assert result.device == img.device


# ============================================================================
# Ultimate Fusion Correctness Tests
# ============================================================================


class TestUltimateFusedCorrectness:
    """Test ultimate fused kernel (geometric + pixel operations)."""
    
    def test_ultimate_matches_sequential_triton(self):
        """Test that ultimate kernel matches sequential triton operations."""
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Parameters
        top, left, height, width = 20, 30, 112, 112
        flip_horizontal = True
        brightness = 1.2
        contrast = 1.1
        saturation = 0.9
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Sequential: geometric fused → pixel fused
        seq_result = F.fused_crop_flip(img, top, left, height, width, flip_horizontal)
        seq_result = F.fused_color_normalize(
            seq_result,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            random_grayscale_p=0.0,
            mean=mean,
            std=std,
        )
        
        # Ultimate fused
        ultimate_result = F.ultimate_fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=flip_horizontal,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            mean=mean,
            std=std,
        )
        
        torch.testing.assert_close(ultimate_result, seq_result)
    
    def test_ultimate_matches_sequential_torchvision(self):
        """Test that ultimate kernel matches sequential torchvision Compose."""
        img = torch.rand(4, 3, 224, 224, device='cuda')
        
        # Parameters
        top, left, height, width = 20, 30, 112, 112
        brightness = 1.2
        saturation = 0.9
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        # Sequential torchvision (no contrast to match exactly)
        tv_result = tvF.crop(img, top, left, height, width)
        tv_result = tvF.horizontal_flip(tv_result)
        tv_result = tvF.adjust_brightness(tv_result, brightness)
        # NOTE: Skipping contrast as torchvision uses blend-with-mean (not fusible)
        tv_result = tvF.adjust_saturation(tv_result, saturation)
        tv_result = tvF.normalize(tv_result, mean, std)
        
        # Triton ultimate fused (without contrast)
        ta_result = F.ultimate_fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=True,
            brightness_factor=brightness,
            contrast_factor=1.0,  # Skip contrast
            saturation_factor=saturation,
            mean=mean,
            std=std,
        )
        
        torch.testing.assert_close(ta_result, tv_result)
    
    @pytest.mark.parametrize("apply_saturation", [False, True])
    def test_ultimate_two_path_optimization(self, apply_saturation):
        """Test that both processing paths produce correct results."""
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        # Parameters
        top, left, height, width = 10, 20, 100, 120
        brightness = 1.1
        contrast = 1.05
        saturation = 0.95 if apply_saturation else 1.0  # 1.0 = skip
        mean = (0.5, 0.5, 0.5)
        std = (0.25, 0.25, 0.25)
        
        # Sequential: crop → flip → color → normalize
        seq_result = F.crop(img, top, left, height, width)
        seq_result = F.horizontal_flip(seq_result)
        seq_result = F.adjust_brightness(seq_result, brightness)
        seq_result = F.adjust_contrast_fast(seq_result, contrast)
        if apply_saturation:
            seq_result = F.adjust_saturation(seq_result, saturation)
        seq_result = F.normalize(seq_result, mean, std)
        
        # Ultimate (will use linear path if saturation=1.0, spatial path otherwise)
        ultimate_result = F.ultimate_fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=True,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            mean=mean,
            std=std,
        )
        
        torch.testing.assert_close(ultimate_result, seq_result)
    
    @pytest.mark.parametrize("shape,crop_size", [
        ((2, 3, 224, 224), (112, 112)),
        ((1, 3, 512, 512), (256, 256)),
        ((8, 3, 128, 128), (64, 64)),
    ])
    def test_ultimate_different_sizes(self, shape, crop_size):
        """Test ultimate kernel with different tensor sizes."""
        img = torch.rand(*shape, device='cuda')
        height, width = crop_size
        
        # Sequential
        seq_result = F.fused_crop_flip(img, 0, 0, height, width, False)
        seq_result = F.fused_color_normalize(
            seq_result,
            brightness_factor=1.1,
            contrast_factor=1.05,
            saturation_factor=0.95,
            random_grayscale_p=0.0,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        # Ultimate
        ultimate_result = F.ultimate_fused_augment(
            img,
            top=0,
            left=0,
            height=height,
            width=width,
            flip_horizontal=False,
            brightness_factor=1.1,
            contrast_factor=1.05,
            saturation_factor=0.95,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        torch.testing.assert_close(ultimate_result, seq_result)


class TestUltimateTransformClass:
    """Test TritonUltimateAugment transform class."""
    
    def test_ultimate_transform_produces_valid_output(self):
        """Test that TritonUltimateAugment produces valid output."""
        from triton_augment.transforms import TritonUltimateAugment
        
        transform = TritonUltimateAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        
        img = torch.rand(4, 3, 224, 224, device='cuda')
        result = transform(img)
        
        assert result.shape == (4, 3, 112, 112)
        assert result.dtype == img.dtype
        assert result.device == img.device
    
    def test_ultimate_transform_deterministic_with_fixed_seed(self):
        """Test that TritonUltimateAugment is deterministic with fixed seed."""
        from triton_augment.transforms import TritonUltimateAugment
        
        transform = TritonUltimateAugment(crop_size=112, brightness=0.2)
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        torch.manual_seed(42)
        result1 = transform(img)
        
        torch.manual_seed(42)
        result2 = transform(img)
        
        torch.testing.assert_close(result1, result2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

