"""
Tests for complete fusion (crop + flip + color + normalize).

Tests the TritonFusedAugment transform and fused_augment functional API
with all operations enabled.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestCompleteFusion:
    """Test complete fusion kernel (geometric + pixel operations)."""
    
    @pytest.mark.parametrize("img_shape,crop_params,flip,brightness,saturation,normalize", [
        # Standard cases - square images
        ((4, 3, 224, 224), (20, 30, 112, 112), True, 1.2, 0.9, True),    # Typical training
        ((2, 3, 256, 256), (0, 0, 128, 128), False, 1.0, 1.0, True),     # Top-left, identity color
        ((1, 3, 512, 512), (100, 100, 256, 256), True, 1.5, 0.5, True),  # Large, extreme colors
        
        # Non-square images
        ((2, 3, 224, 112), (10, 5, 100, 50), False, 0.8, 1.2, True),     # Wide image
        ((3, 3, 112, 224), (5, 20, 50, 100), True, 1.3, 0.7, True),      # Tall image
        ((1, 3, 299, 299), (50, 50, 224, 224), False, 1.1, 0.95, True),  # Inception size
        
        # Odd sizes (not power of 2)
        ((2, 3, 223, 223), (10, 10, 111, 111), True, 1.0, 1.0, True),    # All odd
        ((1, 3, 197, 211), (20, 30, 97, 101), False, 1.4, 0.6, True),    # Prime numbers
        ((4, 3, 225, 225), (25, 25, 175, 175), True, 0.9, 1.1, True),    # Odd multiples
        
        # Small images
        ((8, 3, 64, 64), (8, 8, 48, 48), False, 1.2, 0.8, True),         # Small batch
        ((1, 3, 16, 15), (4, 4, 2, 3), True, 1.5, 1.5, True),          # Tiny image
        ((2, 3, 96, 96), (16, 16, 64, 64), False, 0.7, 0.9, True),       # Small training
        
        # Large batch sizes
        ((16, 3, 128, 128), (16, 16, 96, 96), True, 1.1, 0.95, True),    # Large batch
        ((32, 3, 64, 64), (8, 8, 48, 48), False, 1.0, 1.0, True),        # Very large batch
        
        # Edge case: no crop (full image)
        ((2, 3, 128, 128), (0, 0, 128, 128), True, 1.3, 0.8, True),      # Full image crop
        
        # Edge case: extreme brightness/saturation
        ((1, 3, 224, 224), (50, 50, 112, 112), False, 0.0, 0.0, True),   # Zero brightness & saturation
        ((2, 3, 224, 224), (50, 50, 112, 112), True, 2.0, 2.0, True),    # Max brightness & saturation
        ((1, 3, 1021, 1021), (50, 50, 877, 877), False, 0.5, 1.5, True),   # Dim + saturate, large image
        
        # Edge case: no normalization
        ((2, 3, 224, 224), (20, 30, 112, 112), True, 1.2, 0.9, False),   # Skip normalize
        ((4, 3, 128, 128), (16, 16, 96, 96), False, 1.1, 1.1, False),    # Skip normalize
        
        # Center crops
        ((2, 3, 224, 224), (56, 56, 112, 112), True, 1.0, 1.0, True),    # Perfect center
        ((1, 3, 256, 256), (64, 64, 128, 128), False, 1.25, 0.85, True), # Center crop
        
        # Edge of image crops
        ((2, 3, 224, 224), (112, 112, 112, 112), True, 1.15, 0.92, True),  # Bottom-right corner
        ((1, 3, 320, 320), (0, 160, 160, 160), False, 1.3, 0.7, True),     # Right half
        ((1, 3, 320, 320), (160, 0, 160, 160), True, 0.9, 1.2, True),      # Bottom half
    ])
    def test_fully_fused_matches_torchvision_sequential(self, img_shape, crop_params, flip, brightness, saturation, normalize):
        """
        CRITICAL CORRECTNESS TEST: Complete fusion vs torchvision baseline.
        
        This is the most important test - validates that our fused kernel produces
        identical results to torchvision's sequential operations across a wide
        range of configurations:
        - Various image shapes (square, non-square, odd, small, large)
        - Different crop positions and sizes
        - With/without flip
        - Various brightness/saturation values (including extremes)
        - With/without normalization
        
        NOTE: Skips contrast to ensure exact match (torchvision uses blend-with-mean,
        we use FAST centered scaling).
        """
        img = torch.rand(*img_shape, device='cuda', dtype=torch.float32)
        top, left, height, width = crop_params
        
        # Normalization parameters (ImageNet standard)
        mean = (0.485, 0.456, 0.406) if normalize else None
        std = (0.229, 0.224, 0.225) if normalize else None
        
        # Sequential torchvision (no contrast to match exactly)
        tv_result = tvF.crop(img, top, left, height, width)
        if flip:
            tv_result = tvF.horizontal_flip(tv_result)
        tv_result = tvF.adjust_brightness(tv_result, brightness)
        tv_result = tvF.adjust_saturation(tv_result, saturation)
        if normalize:
            tv_result = tvF.normalize(tv_result, mean=list(mean), std=list(std))
        
        # Triton fused (all in ONE kernel)
        ta_result = F.fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=flip,
            brightness_factor=brightness,
            contrast_factor=1.0,  # Skip contrast (torchvision incompatible)
            saturation_factor=saturation,
            grayscale=False,
            mean=mean,
            std=std,
        )
        
        # Must match exactly!
        torch.testing.assert_close(
            ta_result, 
            tv_result,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Mismatch for shape={img_shape}, crop={crop_params}, flip={flip}, "
                f"brightness={brightness}, saturation={saturation}, normalize={normalize}"
        )
    
    @pytest.mark.parametrize("img_shape,crop_params,brightness,saturation,normalize", [
        # Core saturation cases (most important for grayscale ordering)
        ((2, 3, 224, 224), (20, 30, 112, 112), 1.2, 0.6, True),    # Desaturate
        ((2, 3, 224, 224), (20, 30, 112, 112), 1.2, 1.5, True),    # Oversaturate
        ((2, 3, 224, 224), (20, 30, 112, 112), 1.1, 2.0, True),    # Extreme oversaturation (tests clamp→grayscale)
        ((2, 3, 224, 224), (20, 30, 112, 112), 1.3, 0.0, True),    # Zero saturation
        
        # Size variations
        ((1, 3, 512, 512), (100, 100, 256, 256), 1.2, 0.7, True),  # Large image
        ((1, 3, 16, 15), (2, 2, 12, 11), 1.4, 1.8, True),          # Tiny non-square
        ((8, 3, 64, 64), (8, 8, 48, 48), 1.1, 0.9, True),          # Large batch
        ((1, 3, 1021, 1021), (100, 100, 821, 821), 1.2, 0.8, True),# Large odd size
        
        # Extreme combos
        ((2, 3, 224, 224), (50, 50, 112, 112), 0.0, 0.0, True),    # Both zero
        ((2, 3, 224, 224), (50, 50, 112, 112), 2.0, 2.0, True),    # Both max
        
        # No normalization
        ((2, 3, 224, 224), (20, 30, 112, 112), 1.2, 0.6, False),   # Skip normalize
    ])
    def test_fully_fused_with_grayscale_matches_torchvision(self, img_shape, crop_params, brightness, saturation, normalize):
        """
        CRITICAL TEST: Grayscale ordering correctness vs torchvision.
        
        Validates that grayscale is applied AFTER saturation in the fused kernel,
        matching torchvision's sequential behavior. This is crucial because:
        
        1. Saturation can push values outside [0,1] range
        2. Clamping happens after saturation
        3. Grayscale conversion must use CLAMPED values
        
        If grayscale were applied before saturation (bug we fixed), oversaturated
        images would produce incorrect results.
        
        Tests across:
        - Various image shapes and sizes (square, non-square, small, large)
        - Different saturation factors (0.0 to 3.0, including extreme oversaturation)
        - Various brightness values (0.0 to 2.0)
        - With/without normalization
        - Different crop positions and sizes
        """
        img = torch.rand(*img_shape, device='cuda', dtype=torch.float32)
        top, left, height, width = crop_params
        
        # Normalization parameters (ImageNet standard)
        mean = (0.485, 0.456, 0.406) if normalize else None
        std = (0.229, 0.224, 0.225) if normalize else None
        
        # Apply fused with grayscale=True (force grayscale for all images)
        # Note: Uses FAST contrast (centered scaling), not torchvision contrast
        ta_result = F.fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=True,
            brightness_factor=brightness,
            contrast_factor=1.0,  # Skip contrast since we use FAST version
            saturation_factor=saturation,
            grayscale=True,  # Force grayscale
            mean=mean,
            std=std
        )
        
        # Apply torchvision sequential operations
        # 1. Crop
        tv_result = tvF.crop(img, top, left, height, width)
        # 2. Horizontal flip
        tv_result = tvF.horizontal_flip(tv_result)
        # 3. Brightness
        tv_result = tvF.adjust_brightness(tv_result, brightness)
        # 4. Saturation (critical: happens BEFORE grayscale!)
        tv_result = tvF.adjust_saturation(tv_result, saturation)
        # 5. Grayscale (convert to grayscale with 3 output channels)
        tv_result = tvF.rgb_to_grayscale(tv_result, num_output_channels=3)
        # 6. Normalize
        if normalize:
            tv_result = tvF.normalize(tv_result, mean=list(mean), std=list(std))
        
        # Must match exactly - validates correct saturation→clamp→grayscale ordering!
        torch.testing.assert_close(
            ta_result, 
            tv_result, 
            rtol=1e-5, 
            atol=1e-5,
            msg=f"Grayscale ordering mismatch for shape={img_shape}, crop={crop_params}, "
                f"brightness={brightness}, saturation={saturation}, normalize={normalize}"
        )
    
    @pytest.mark.parametrize("apply_saturation", [False, True])
    def test_kernel_linear_vs_spatial_path_correctness(self, apply_saturation):
        """
        Test that kernel's two internal paths (linear vs spatial) produce correct results.
        
        The kernel uses a linear path when saturation=1.0 (skip) and a spatial
        path when saturation≠1.0. This validates both code paths.
        """
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
        
        # Fused (will use linear path if saturation=1.0, spatial path otherwise)
        fused_result = F.fused_augment(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            flip_horizontal=True,
            brightness_factor=brightness,
            contrast_factor=contrast,
            saturation_factor=saturation,
            grayscale=False,
            mean=mean,
            std=std,
        )
        
        torch.testing.assert_close(fused_result, seq_result)
    
    @pytest.mark.parametrize("shape,crop_size", [
        ((2, 3, 224, 224), (112, 112)),
        ((1, 3, 512, 512), (256, 256)),
        ((8, 3, 128, 128), (64, 64)),
    ])
    def test_fully_fused_matches_two_pass_fusion_across_sizes(self, shape, crop_size):
        """
        Test that single-pass complete fusion matches two-pass fusion across sizes.
        
        Verifies that doing all operations in ONE kernel produces the same
        result as doing geometric ops in one kernel, then color ops in another.
        Tests across multiple image/crop sizes for robustness.
        """
        img = torch.rand(*shape, device='cuda')
        height, width = crop_size
        
        # Sequential (both steps using ultimate kernel)
        seq_result = F.fused_augment(
            img, 0, 0, height, width, False,
            brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0,  # No-op color in step 1
            grayscale=False, mean=None, std=None
        )
        _, _, h, w = seq_result.shape
        seq_result = F.fused_augment(
            seq_result,
            top=0, left=0, height=h, width=w, flip_horizontal=False,  # No-op geometric in step 2
            brightness_factor=1.1,
            contrast_factor=1.05,
            saturation_factor=0.95,
            grayscale=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        # Fused
        fused_result = F.fused_augment(
            img,
            top=0,
            left=0,
            height=height,
            width=width,
            flip_horizontal=False,
            brightness_factor=1.1,
            contrast_factor=1.05,
            saturation_factor=0.95,
            grayscale=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        torch.testing.assert_close(fused_result, seq_result)


class TestFusedAugmentTransform:
    """Test TritonFusedAugment transform class."""
    
    def test_fused_augment_produces_valid_output(self):
        """Test that TritonFusedAugment produces valid output."""
        transform = ta.TritonFusedAugment(
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
    
    def test_fused_augment_deterministic_with_fixed_seed(self):
        """Test that TritonFusedAugment is deterministic with fixed seed."""
        transform = ta.TritonFusedAugment(crop_size=112, brightness=0.2)
        img = torch.rand(2, 3, 224, 224, device='cuda')
        
        torch.manual_seed(42)
        result1 = transform(img)
        
        torch.manual_seed(42)
        result2 = transform(img)
        
        torch.testing.assert_close(result1, result2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

