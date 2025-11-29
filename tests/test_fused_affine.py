
import torch
import pytest
import triton_augment as ta
import torchvision.transforms.v2.functional as TVF
from triton_augment.functional import InterpolationMode

@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("interpolation", ["nearest", "bilinear"])
@pytest.mark.parametrize("imgsize", [(100, 111), (256, 256)])
@pytest.mark.parametrize("shear", [[0.0, 0.0], [5, 1]])
@pytest.mark.parametrize("flip", [True, False])
def test_fused_affine_vs_sequential(batch_size, interpolation, imgsize, shear, flip):
    """
    Verify that TritonFusedAffineAugment matches sequential application of transforms.
    We test the geometric composition: Affine -> Crop -> Flip.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = "cuda"
    # Create a random image
    img = torch.rand(batch_size, 3, imgsize[0], imgsize[1], device=device)
    
    # 1. Define transforms
    angle = 30.0
    dx, dy = 10.0, 5.0
    scale_factor = 1.1
    crop_top, crop_left = 10, 10
    crop_h, crop_w = 80, 80
    
    # 2. Sequential execution (Ground Truth)
    # Note: Our fused pipeline does: Affine -> Crop -> Flip
    # So we must match that order.
    
    # Step A: Affine
    # Torchvision affine rotates around center.
    # We need to match the center logic.
    center = [img.shape[-1] * 0.5, img.shape[-2] * 0.5]
    
    out_seq = TVF.affine(
        img, 
        angle=angle, 
        translate=[dx, dy], 
        scale=scale_factor, 
        shear=shear,
        interpolation=TVF.InterpolationMode.NEAREST if interpolation == "nearest" else TVF.InterpolationMode.BILINEAR,
        center=center
    )
    
    # Step B: Crop
    out_seq = TVF.crop(out_seq, crop_top, crop_left, crop_h, crop_w)
    
    # Step C: Flip
    if flip:
        out_seq = TVF.hflip(out_seq)
        
    # 3. Fused execution via fused_augment (New API)
    # The new API handles matrix construction internally.
    
    # Run Fused Kernel
    out_fused = ta.functional.fused_augment(
        img,
        # Crop params
        top=crop_top,
        left=crop_left,
        height=crop_h,
        width=crop_w,
        flip_horizontal=flip,
        # Affine params
        angle=angle,
        translate=[dx, dy],
        scale=scale_factor,
        shear=shear,
        interpolation=interpolation,
        center=center
    )
    
    # Compare
    # Note: Nearest neighbor can have off-by-one errors due to rounding differences
    # between composed matrix and sequential integer ops.
    # Bilinear should be very close.
    
    if interpolation == "nearest":
        # Allow some mismatch pixels
        diff = (out_fused - out_seq).abs()
        mismatch_pct = (diff > 1e-4).float().mean().item()
        print(f"Nearest Mismatch: {mismatch_pct:.2%}")
        assert mismatch_pct < 0.05 # Allow <5% mismatch for complex composition
    else:
        # Bilinear should be closer
        diff = (out_fused - out_seq).abs()
        mae = diff.mean().item()
        print(f"Bilinear MAE: {mae:.4f}")
        assert mae < 0.05

if __name__ == "__main__":
    test_fused_affine_vs_sequential(1, "nearest")
    test_fused_affine_vs_sequential(1, "bilinear")
