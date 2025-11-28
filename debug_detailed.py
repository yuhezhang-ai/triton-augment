import torch
import triton_augment as ta
import triton_augment.functional as F
import torchvision.transforms.v2.functional as tvF

# Reproduce the failing test case
# test_affine_matches_torchvision[nearest-15.0-translate4-1.0-shear4-1-224-224]
batch_size = 8  # batch_size parameter from test
height = 111    # height parameter from test
width = 100     # width parameter from test
angle = 90.0    # 90 degree rotation
translate = [0.0, 0.0]  # translate2
scale = 1.0
shear = [0.0, 0.0]  # shear2

# Create test image
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

# Print middle row (y=112) for channel 0
print(f"\n=== Middle row (y=112, channel=0) ===")
y = 112
print(f"{'x':<5} {'TV':<12} {'TA':<12} {'Diff':<12} {'Match'}")
print("-" * 50)

for x in range(0, width, 10):  # Sample every 10 pixels
    tv_val = tv_result[0, 0, y, x].item()
    ta_val = ta_result[0, 0, y, x].item()
    diff_val = abs(tv_val - ta_val)
    match = "✓" if diff_val < 1e-3 else "✗"
    print(f"{x:<5} {tv_val:<12.6f} {ta_val:<12.6f} {diff_val:<12.6f} {match}")

# Also check some edge pixels
print(f"\n=== Edge pixels (channel=0) ===")
edge_coords = [
    (0, 0, "top-left"),
    (0, width-1, "top-right"),
    (height-1, 0, "bottom-left"),
    (height-1, width-1, "bottom-right"),
    (height//2, 0, "middle-left"),
    (height//2, width-1, "middle-right"),
]

print(f"{'Location':<15} {'(y,x)':<12} {'TV':<12} {'TA':<12} {'Diff':<12} {'Match'}")
print("-" * 70)

for y, x, label in edge_coords:
    tv_val = tv_result[0, 0, y, x].item()
    ta_val = ta_result[0, 0, y, x].item()
    diff_val = abs(tv_val - ta_val)
    match = "✓" if diff_val < 1e-3 else "✗"
    print(f"{label:<15} ({y},{x}){'':<6} {tv_val:<12.6f} {ta_val:<12.6f} {diff_val:<12.6f} {match}")

# Find clusters of mismatches
print(f"\n=== Mismatch distribution ===")
diff_map = (diff[0, 0] > 1e-3).cpu().numpy()
import numpy as np

# Count mismatches per row
row_mismatches = diff_map.sum(axis=1)
print(f"Rows with most mismatches:")
top_rows = np.argsort(row_mismatches)[-5:][::-1]
for row in top_rows:
    print(f"  Row {row}: {row_mismatches[row]} mismatches")

# Count mismatches per column
col_mismatches = diff_map.sum(axis=0)
print(f"\nColumns with most mismatches:")
top_cols = np.argsort(col_mismatches)[-5:][::-1]
for col in top_cols:
    print(f"  Col {col}: {col_mismatches[col]} mismatches")
