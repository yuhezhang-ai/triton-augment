import torch
import triton_augment.functional as F
import math

# Parameters from the failing test
batch_size = 1  # Just check first sample
height = 111
width = 100
angle = 15.0
translate = [5.0, 5.0]
scale = 1.0
shear = [10.0, -5.0]

# Mismatch location from debug output: [0,0,33,97]
x_out = 97
y_out = 33

# Calculate the affine matrix
from triton_augment.functional import _get_inverse_affine_matrix

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

a, b, c_tx, d, e, f_ty = matrix[0].tolist()

print(f"Matrix: [{a:.6f}, {b:.6f}, {c_tx:.6f}, {d:.6f}, {e:.6f}, {f_ty:.6f}]")
print(f"\nOutput pixel: ({x_out}, {y_out})")

# Simulate kernel coordinate calculation
half_w = width * 0.5
half_h = height * 0.5
x_centered = x_out - half_w + 0.5
y_centered = y_out - half_h + 0.5

print(f"Centered: ({x_centered:.6f}, {y_centered:.6f})")

x_norm = (a * x_centered + b * y_centered + c_tx) / half_w
y_norm = (d * x_centered + e * y_centered + f_ty) / half_h

print(f"Normalized: ({x_norm:.6f}, {y_norm:.6f})")

x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
y_in = ((y_norm + 1.0) * height - 1.0) * 0.5

print(f"Input coords: ({x_in:.10f}, {y_in:.10f})")

# Check floor and fraction
x_floor = math.floor(x_in)
y_floor = math.floor(y_in)
x_frac = x_in - x_floor
y_frac = y_in - y_floor

print(f"Floor: ({x_floor}, {y_floor})")
print(f"Frac: ({x_frac:.10f}, {y_frac:.10f})")
print(f"Diff from 0.5: ({abs(x_frac - 0.5):.10e}, {abs(y_frac - 0.5):.10e})")

# What would RNE do?
def round_rne_python(val, epsilon=1e-4):
    val_floor = math.floor(val)
    val_frac = val - val_floor
    
    is_half = abs(val_frac - 0.5) < epsilon
    
    if is_half:
        # Snap to even
        val_floor_int = int(val_floor)
        is_odd = (val_floor_int % 2) != 0
        return val_floor_int + (1 if is_odd else 0)
    else:
        # Standard round
        return int(math.floor(val + 0.5))

x_rne = round_rne_python(x_in, 1e-4)
y_rne = round_rne_python(y_in, 1e-4)

print(f"\nRNE result (1e-4): ({x_rne}, {y_rne})")

# What would PyTorch do?
x_torch_round = torch.round(x_in)
y_torch_round = torch.round(y_in)

print(f"Python round(): ({x_torch_round}, {y_torch_round})")

# Check if they differ
if x_rne != x_torch_round or y_rne != y_torch_round:
    print(f"\n⚠️  MISMATCH! RNE != round()")
else:
    print(f"\n✓ Match")
