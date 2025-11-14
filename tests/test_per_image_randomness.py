"""
Tests for per-image randomness feature.

Tests that same_on_batch parameter works correctly in transform classes
and that functional API accepts per-image tensor parameters.
"""

import pytest
import torch
import triton_augment as ta
import triton_augment.functional as F

try:
    import torchvision.transforms.v2.functional as tvF
except ImportError:
    tvF = None


class TestPerImageRandomnessTransforms:
    """Test per-image randomness in transform classes."""
    
    def test_fused_augment_per_image_produces_different_results(self):
        """Test that same_on_batch=False produces different augmentations per image."""
        transform = ta.TritonFusedAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=(0.5, 1.5),
            saturation=(0.5, 1.5),
            same_on_batch=False
        )
        
        # Create identical images in batch
        img = torch.rand(4, 3, 224, 224, device='cuda')
        identical_batch = img[0:1].expand(4, -1, -1, -1).clone()
        
        torch.manual_seed(42)
        result = transform(identical_batch)
        
        # Check that not all images in batch are identical (per-image randomness worked)
        # Compare first image with others
        for i in range(1, 4):
            max_diff = torch.abs(result[0] - result[i]).max().item()
            # At least one image should be different (very high probability with random crop/flip/color)
            if max_diff > 0.01:
                break
        else:
            # If we get here, all images are identical (should not happen)
            pytest.fail("All images in batch are identical despite same_on_batch=False")
    
    def test_fused_augment_per_image_false_produces_same_results(self):
        """Test that same_on_batch=True produces identical augmentations."""
        transform = ta.TritonFusedAugment(
            crop_size=112,
            horizontal_flip_p=0.5,
            brightness=(0.5, 1.5),
            saturation=(0.5, 1.5),
            same_on_batch=True
        )
        
        # Create identical images in batch
        img = torch.rand(1, 3, 224, 224, device='cuda')
        identical_batch = img.expand(4, -1, -1, -1).clone()
        
        torch.manual_seed(42)
        result = transform(identical_batch)
        
        # All images should be identical (same augmentation applied)
        for i in range(1, 4):
            torch.testing.assert_close(result[0], result[i])
    
    def test_random_crop_per_image_produces_different_crops(self):
        """Test that TritonRandomCrop with same_on_batch=False produces different crops."""
        transform = ta.TritonRandomCrop(112, same_on_batch=False)
        
        # Create images with unique gradients so we can detect different crop positions
        img = torch.zeros(4, 3, 224, 224, device='cuda')
        # Each image has a unique linear gradient
        for i in range(4):
            # Create a unique gradient pattern for each image
            gradient = torch.linspace(0, 1, 224, device='cuda')
            img[i, 0, :, :] = gradient.unsqueeze(1)  # Vertical gradient
            img[i, 1, :, :] = gradient.unsqueeze(0)  # Horizontal gradient
            img[i, 2, :, :] = (i + 1) * 0.25  # Unique constant per image
        
        torch.manual_seed(42)
        result = transform(img)
        
        # Check that crops are at different positions by comparing the mean values
        # Different crop positions will have different mean values due to the gradients
        means = [result[i].mean().item() for i in range(4)]
        
        # With random cropping and per-image randomness, we expect different means
        # (extremely unlikely to get identical means from different random crops)
        assert len(set([round(m, 6) for m in means])) > 1, \
            f"All crops have identical means {means}, suggesting same crop position for all images"
    
    def test_random_horizontal_flip_per_image(self):
        """Test that TritonRandomHorizontalFlip with same_on_batch flips differently."""
        transform = ta.TritonRandomHorizontalFlip(p=0.5, same_on_batch=False)
        
        # Create batch with asymmetric pattern (detectable flip)
        img = torch.zeros(8, 3, 64, 64, device='cuda')
        img[:, :, :, 0:32] = 1.0  # Left half bright
        
        torch.manual_seed(42)
        result = transform(img)
        
        # Check if some images are flipped and some are not
        # Flipped: right half bright, Not flipped: left half bright
        left_bright = (result[:, 0, 32, 10] > 0.5).sum().item()
        right_bright = (result[:, 0, 32, 50] > 0.5).sum().item()
        
        # With p=0.5 and 8 images, we expect some flipped and some not (statistically)
        assert left_bright != 8 and right_bright != 8, "All images treated identically"
    
    def test_random_grayscale_per_image(self):
        """Test that TritonRandomGrayscale with same_on_batch works per-image."""
        transform = ta.TritonRandomGrayscale(p=0.5, same_on_batch=False)
        
        img = torch.rand(8, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        torch.manual_seed(42)
        result = transform(img)
        
        # Check how many images are grayscale
        grayscale_count = 0
        for i in range(8):
            if torch.allclose(result[i, 0], result[i, 1], atol=1e-5):
                grayscale_count += 1
        
        # With p=0.5 and 8 images, we expect some grayscale and some not (statistically)
        assert 0 < grayscale_count < 8, f"Expected mixed grayscale/color, got {grayscale_count}/8 grayscale"


class TestPerImageFunctionalAPI:
    """Test functional API with per-image tensor parameters."""
    
    def test_color_jitter_per_image_matches_torchvision_per_image(self):
        """Test that per-image color jitter matches per-image torchvision application."""
        img = torch.rand(4, 3, 224, 224, device='cuda', dtype=torch.float32)
        
        # Manually generate random parameters with fixed seed
        torch.manual_seed(42)
        brightness_factors = torch.empty(4, device='cuda').uniform_(0.8, 1.2)
        saturation_factors = torch.empty(4, device='cuda').uniform_(0.8, 1.2)
        
        # Apply triton with per-image parameters
        ta_result = F.fused_augment(
            img,
            top=0, left=0, height=224, width=224, flip_horizontal=False,  # No-op geometric
            brightness_factor=brightness_factors,
            contrast_factor=1.0,  # Skip contrast (FAST mode differs)
            saturation_factor=saturation_factors,
            grayscale=False,
        )
        
        # Apply torchvision per-image (loop)
        tv_result = []
        for i in range(4):
            img_i = tvF.adjust_brightness(img[i:i+1], brightness_factors[i].item())
            img_i = tvF.adjust_saturation(img_i, saturation_factors[i].item())
            tv_result.append(img_i)
        tv_result = torch.cat(tv_result, dim=0)
        
        # Should match
        torch.testing.assert_close(ta_result, tv_result, rtol=1e-5, atol=1e-5)
    
    def test_crop_flip_per_image_matches_torchvision(self):
        """Test that per-image crop+flip matches per-image torchvision application."""
        img = torch.rand(4, 3, 224, 224, device='cuda', dtype=torch.float32)
        
        # Manual per-image parameters
        top_offsets = torch.tensor([10, 20, 30, 40], device='cuda', dtype=torch.int32)
        left_offsets = torch.tensor([15, 25, 35, 45], device='cuda', dtype=torch.int32)
        flip_mask = torch.tensor([0, 1, 0, 1], device='cuda', dtype=torch.uint8)
        height, width = 100, 120
        
        # Apply triton with per-image parameters
        ta_result = F.fused_augment(
            img,
            top=top_offsets,
            left=left_offsets,
            height=height,
            width=width,
            flip_horizontal=flip_mask,
            brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0,  # No-op color
            grayscale=False, mean=None, std=None  # No-op
        )
        
        # Apply torchvision per-image
        tv_result = []
        for i in range(4):
            img_i = tvF.crop(img[i:i+1], top_offsets[i].item(), left_offsets[i].item(), height, width)
            if flip_mask[i].item() == 1:
                img_i = tvF.horizontal_flip(img_i)
            tv_result.append(img_i)
        tv_result = torch.cat(tv_result, dim=0)
        
        # Should match exactly
        torch.testing.assert_close(ta_result, tv_result)
    
    def test_grayscale_per_image_mask(self):
        """Test that per-image grayscale mask works correctly."""
        img = torch.rand(4, 3, 128, 128, device='cuda', dtype=torch.float32)
        
        # Grayscale mask: convert images 0 and 2, keep 1 and 3 as RGB
        grayscale_mask = torch.tensor([1, 0, 1, 0], device='cuda', dtype=torch.uint8)
        
        result = F.rgb_to_grayscale(img, num_output_channels=3, grayscale_mask=grayscale_mask)
        
        # Images 0 and 2 should be grayscale
        assert torch.allclose(result[0, 0], result[0, 1])
        assert torch.allclose(result[0, 1], result[0, 2])
        assert torch.allclose(result[2, 0], result[2, 1])
        assert torch.allclose(result[2, 1], result[2, 2])
        
        # Images 1 and 3 should be unchanged
        torch.testing.assert_close(result[1], img[1])
        torch.testing.assert_close(result[3], img[3])
    
    @pytest.mark.parametrize("batch_size,img_size,crop_size,use_normalize,use_grayscale", [
        # Core cases
        (4, 224, 112, True, False),     # Standard training with normalize
        (2, 256, 128, False, False),    # No normalize
        (8, 128, 96, True, False),      # Large batch
        
        # With grayscale
        (4, 224, 183, True, True),      # With grayscale + normalize
        (2, 225, 112, False, True),     # With grayscale, no normalize
        
        # Size variations
        (1, 1021, 877, True, False),     # Large image, single item
        (16, 64, 48, True, False),      # Small images, large batch
        (4, 299, 223, True, False),     # Inception size, odd size
    ])
    def test_fully_fused_per_image_matches_torchvision(self, batch_size, img_size, crop_size, use_normalize, use_grayscale):
        """
        CRITICAL TEST: Fully fused kernel with per-image parameters vs torchvision.
        
        Validates that the fused kernel with per-image random parameters produces
        identical results to applying torchvision operations per-image in a loop.
        
        This ensures that:
        1. Per-image crop offsets work correctly
        2. Per-image flip decisions work correctly
        3. Per-image color adjustments work correctly
        4. Per-image grayscale decisions work correctly
        5. All operations are correctly ordered and applied
        """
        torch.manual_seed(42)
        img = torch.rand(batch_size, 3, img_size, img_size, device='cuda', dtype=torch.float32)
        
        # Generate per-image random parameters
        torch.manual_seed(100)
        max_offset = img_size - crop_size
        top_offsets = torch.randint(0, max_offset + 1, (batch_size,), device='cuda', dtype=torch.int32)
        left_offsets = torch.randint(0, max_offset + 1, (batch_size,), device='cuda', dtype=torch.int32)
        flip_mask = torch.randint(0, 2, (batch_size,), device='cuda', dtype=torch.uint8)
        
        brightness_factors = torch.empty(batch_size, device='cuda').uniform_(0.8, 1.2)
        saturation_factors = torch.empty(batch_size, device='cuda').uniform_(0.7, 1.3)
        
        grayscale_mask = None
        if use_grayscale:
            grayscale_mask = torch.randint(0, 2, (batch_size,), device='cuda', dtype=torch.uint8)
        
        mean = (0.485, 0.456, 0.406) if use_normalize else None
        std = (0.229, 0.224, 0.225) if use_normalize else None
        
        # Apply Triton fully fused (ALL operations in ONE kernel with per-image params)
        ta_result = F.fused_augment(
            img,
            top=top_offsets,
            left=left_offsets,
            height=crop_size,
            width=crop_size,
            flip_horizontal=flip_mask,
            brightness_factor=brightness_factors,
            contrast_factor=1.0,  # Skip contrast (FAST mode differs from torchvision)
            saturation_factor=saturation_factors,
            grayscale=grayscale_mask if use_grayscale else False,
            mean=mean,
            std=std,
        )
        
        # Apply torchvision per-image (loop through batch)
        tv_result = []
        for i in range(batch_size):
            # 1. Crop
            img_i = tvF.crop(
                img[i:i+1],
                top_offsets[i].item(),
                left_offsets[i].item(),
                crop_size,
                crop_size
            )
            
            # 2. Horizontal flip
            if flip_mask[i].item() == 1:
                img_i = tvF.horizontal_flip(img_i)
            
            # 3. Brightness
            img_i = tvF.adjust_brightness(img_i, brightness_factors[i].item())
            
            # 4. Saturation
            img_i = tvF.adjust_saturation(img_i, saturation_factors[i].item())
            
            # 5. Grayscale (if enabled for this image)
            if use_grayscale and grayscale_mask[i].item() == 1:
                img_i = tvF.rgb_to_grayscale(img_i, num_output_channels=3)
            
            # 6. Normalize
            if use_normalize:
                img_i = tvF.normalize(img_i, mean=list(mean), std=list(std))
            
            tv_result.append(img_i)
        
        tv_result = torch.cat(tv_result, dim=0)
        
        # Must match exactly - validates per-image correctness!
        torch.testing.assert_close(
            ta_result,
            tv_result,
            rtol=1e-5,
            atol=1e-5,
            msg=f"Per-image mismatch for batch_size={batch_size}, img_size={img_size}, "
                f"crop_size={crop_size}, normalize={use_normalize}, grayscale={use_grayscale}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

