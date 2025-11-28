import torch
import triton_augment as ta
import triton_augment.functional as F
import torchvision.transforms.v2.functional as tvF

# Reproduce the failing test case
# test_affine_matches_torchvision[nearest-90.0-translate2-1.0-shear2-8-111-100]
batch_size = 8  # batch_size parameter from test
height = 111    # height parameter from test
width = 100     # width parameter from test
angle = 90.0    # 90 degree rotation
translate = [0.0, 0.0]  # translate2
scale = 1.0
shear = [0.0, 0.0]  # shear2

# Create test image (no seed set, matching the test)
img = torch.rand(batch_size, 3, height, width, device='cuda')

# Get interpolation modes
tv_interp = tvF.InterpolationMode.NEAREST
ta_interp = ta.InterpolationMode.NEAREST

# Torchvision affine
tv_result = tvF.affine(
    img, angle=angle, translate=translate, scale=scale, shear=shear,
    interpolation=tv_interp
)

# Triton affine
ta_result = F.affine(
    img, angle=angle, translate=translate, scale=scale, shear=shear,
    interpolation=ta_interp
)

# Compare
diff = (tv_result - ta_result).abs()
max_diff = diff.max()
num_diffs = (diff > 1e-3).sum()
total = diff.numel()

print(f"Max diff: {max_diff}")
print(f"Num diffs: {num_diffs} / {total} ({100.0 * num_diffs / total:.2f}%)")

# Find some specific mismatches
indices = torch.nonzero(diff > 1e-3)
print(f"\nFirst 10 mismatches:")
for idx in indices[:10]:
    b, c, y, x = idx.tolist()
    print(f"  [{b},{c},{y},{x}]: TV={tv_result[b,c,y,x]:.6f}, TA={ta_result[b,c,y,x]:.6f}, diff={diff[b,c,y,x]:.6f}")

# Check the affine matrix
from triton_augment.functional import _get_inverse_affine_matrix

# Convert parameters to tensors
center_tensor = torch.zeros(batch_size, 2, device='cuda', dtype=torch.float32)
angle_tensor = torch.full((batch_size,), angle, device='cuda', dtype=torch.float32)
translate_tensor = torch.tensor([translate], device='cuda', dtype=torch.float32).repeat(batch_size, 1)
scale_tensor = torch.full((batch_size,), scale, device='cuda', dtype=torch.float32)
shear_tensor = torch.tensor([shear], device='cuda', dtype=torch.float32).repeat(batch_size, 1)

matrix = _get_inverse_affine_matrix(
    center_tensor,
    angle_tensor,
    translate_tensor,
    scale_tensor,
    shear_tensor
)

print(f"\nAffine matrix (first sample):")
print(matrix[0])
print(f"Matrix min/max: {matrix.min():.10f} / {matrix.max():.10f}")
print(f"Matrix abs min (non-zero): {matrix[matrix != 0].abs().min():.10e}")
