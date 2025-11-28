import torch
import math

# 90-degree rotation parameters
height = 111
width = 100
angle = 90.0

# Matrix for 90-degree rotation (from debug output)
a, b, c_tx = 0.0, 1.0, 0.0
d, e, f_ty = -1.0, 0.0, 0.0

print("=== 90-Degree Rotation Coordinate Analysis ===")
print(f"Image size: {height}x{width}")
print(f"Matrix: [{a}, {b}, {c_tx}, {d}, {e}, {f_ty}]")
print()

# Test a few specific output pixels
test_pixels = [
    (5, 2),   # From the mismatch list
    (5, 4),
    (50, 55), # Middle-ish
    (0, 0),   # Corner
    (width-1, height-1),  # Opposite corner
]

half_w = width * 0.5
half_h = height * 0.5

print(f"half_w = {half_w}, half_h = {half_h}")
print()

for x_out, y_out in test_pixels:
    print(f"Output pixel: ({x_out}, {y_out})")
    
    # Step 1: Convert to centered coordinates
    x_centered = x_out - half_w + 0.5
    y_centered = y_out - half_h + 0.5
    print(f"  Centered: ({x_centered:.6f}, {y_centered:.6f})")
    
    # Step 2: Apply matrix with rescaling
    x_norm = (a * x_centered + b * y_centered + c_tx) / half_w
    y_norm = (d * x_centered + e * y_centered + f_ty) / half_h
    print(f"  Normalized: ({x_norm:.10f}, {y_norm:.10f})")
    
    # Step 3: Convert to input pixel coords
    x_in = ((x_norm + 1.0) * width - 1.0) * 0.5
    y_in = ((y_norm + 1.0) * height - 1.0) * 0.5
    print(f"  Input coords: ({x_in:.10f}, {y_in:.10f})")
    
    # Check floor and fraction
    x_floor = math.floor(x_in)
    y_floor = math.floor(y_in)
    x_frac = x_in - x_floor
    y_frac = y_in - y_floor
    
    print(f"  Floor: ({x_floor}, {y_floor})")
    print(f"  Frac: ({x_frac:.10f}, {y_frac:.10f})")
    print(f"  Diff from 0.5: ({abs(x_frac - 0.5):.10e}, {abs(y_frac - 0.5):.10e})")
    
    # What would RNE do with epsilon=1e-4?
    def round_rne(val, epsilon=1e-4):
        val_floor = math.floor(val)
        val_frac = val - val_floor
        is_half = abs(val_frac - 0.5) < epsilon
        
        if is_half:
            val_floor_int = int(val_floor)
            is_odd = (val_floor_int % 2) != 0
            return val_floor_int + (1 if is_odd else 0)
        else:
            return int(math.floor(val + 0.5))
    
    x_rne = round_rne(x_in)
    y_rne = round_rne(y_in)
    
    # What would Python round() do?
    x_py_round = round(x_in)
    y_py_round = round(y_in)
    
    print(f"  RNE (1e-4): ({x_rne}, {y_rne})")
    print(f"  Python round(): ({x_py_round}, {y_py_round})")
    
    if x_rne != x_py_round or y_rne != y_py_round:
        print(f"  ⚠️  MISMATCH!")
    else:
        print(f"  ✓ Match")
    
    print()
