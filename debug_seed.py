"""
Debug script to find why transform() and _get_params() produce different results on CUDA.
"""
import torch
import triton_augment as ta
import triton_augment.functional as F

batch_size = 4
height, width = 128, 128

transform = ta.TritonRandomAffine(
    degrees=45,
    translate=(0.2, 0.2),
    scale=(0.8, 1.2),
    shear=15,
    interpolation=ta.InterpolationMode.BILINEAR,
    same_on_batch=False
)

# Create image with different seed
torch.manual_seed(123)
img = torch.rand(batch_size, 3, height, width, device='cuda')

print("=== Test 1: Call transform() with seed 42 ===")
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # Also set CUDA seed explicitly
transform_result = transform(img)
print(f"Transform result shape: {transform_result.shape}")
print(f"Transform result[0,0,0,:5]: {transform_result[0,0,0,:5].tolist()}")

print("\n=== Test 2: Call _get_params with seed 42 ===")
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # Also set CUDA seed explicitly
angle, translate, scale, shear = transform._get_params(batch_size, img.device, (height, width))
print(f"Angle: {angle.tolist()}")
print(f"Translate: {translate.tolist()}")
print(f"Scale: {scale.tolist()}")
print(f"Shear: {shear.tolist()}")

print("\n=== Test 3: Apply F.affine with those params ===")
functional_result = F.affine(
    img,
    angle=angle,
    translate=translate,
    scale=scale,
    shear=shear,
    interpolation=ta.InterpolationMode.BILINEAR
)
print(f"Functional result[0,0,0,:5]: {functional_result[0,0,0,:5].tolist()}")

print("\n=== Test 4: Compare ===")
diff = (transform_result - functional_result).abs()
print(f"Max diff: {diff.max().item()}")
print(f"Mean diff: {diff.mean().item()}")
print(f"Match: {torch.allclose(transform_result, functional_result)}")

print("\n=== Test 5: Get params that transform actually used ===")
# Run transform again and capture what params it generates
torch.manual_seed(42)
torch.cuda.manual_seed(42)
angle2, translate2, scale2, shear2 = transform._get_params(batch_size, img.device, (height, width))
print(f"Angle (from _get_params after seed 42): {angle2.tolist()}")

# Now run transform
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Manually trace through forward()
from triton_augment.transforms import _normalize_video_shape, _compute_param_count
normalized_img, bs, nf, orig_shape, was_3d = _normalize_video_shape(img)
param_count = _compute_param_count(bs, nf, transform.same_on_batch, transform.same_on_frame)
print(f"param_count from forward logic: {param_count}")
angle3, translate3, scale3, shear3 = transform._get_params(param_count, img.device, (height, width))
print(f"Angle (from simulated forward): {angle3.tolist()}")

print(f"\nAngles match: {torch.allclose(angle2, angle3)}")
