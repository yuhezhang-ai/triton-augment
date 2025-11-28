import torch

# Replicate PyTorch's _affine_grid function
def my_affine_grid(theta, w, h, ow, oh):
    """
    Replicate PyTorch's _affine_grid from torchvision
    theta: [1, 2, 3] affine matrix
    w, h: input width, height
    ow, oh: output width, height
    """
    dtype = theta.dtype
    device = theta.device
    
    # Create base_grid
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    
    # x_grid: from (1-ow)*0.5 to (ow-1)*0.5
    x_grid = torch.linspace((1.0 - ow) * 0.5, (ow - 1.0) * 0.5, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    
    # y_grid: from (1-oh)*0.5 to (oh-1)*0.5
    y_grid = torch.linspace((1.0 - oh) * 0.5, (oh - 1.0) * 0.5, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    
    base_grid[..., 2].fill_(1)
    
    # Rescale theta
    rescaled_theta = theta.transpose(1, 2).div_(torch.tensor([0.5 * w, 0.5 * h], dtype=dtype, device=device))
    
    # Apply transformation
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    
    return output_grid.view(1, oh, ow, 2)

# Test with 90-degree rotation, 111x100
height, width = 111, 100
theta = torch.tensor([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0]
], dtype=torch.float32, device='cuda').unsqueeze(0)

# My implementation
my_grid = my_affine_grid(theta, w=width, h=height, ow=width, oh=height)

# PyTorch's implementation
import torch.nn.functional as F
pytorch_grid = F.affine_grid(theta, [1, 1, height, width], align_corners=False)

# Compare
print("=== Comparing my_affine_grid vs PyTorch's affine_grid ===\n")

test_pixels = [(5, 2), (5, 4), (50, 55)]

for x, y in test_pixels:
    my_gx = my_grid[0, y, x, 0].item()
    my_gy = my_grid[0, y, x, 1].item()
    
    pt_gx = pytorch_grid[0, y, x, 0].item()
    pt_gy = pytorch_grid[0, y, x, 1].item()
    
    print(f"Pixel ({x}, {y}):")
    print(f"  My grid: ({my_gx:.10f}, {my_gy:.10f})")
    print(f"  PyTorch: ({pt_gx:.10f}, {pt_gy:.10f})")
    print(f"  Diff: ({abs(my_gx - pt_gx):.10e}, {abs(my_gy - pt_gy):.10e})")
    
    if abs(my_gx - pt_gx) < 1e-6 and abs(my_gy - pt_gy) < 1e-6:
        print("  ✓ Match!")
    else:
        print("  ✗ Mismatch!")
    print()

# Now let's manually calculate what our kernel does for pixel (5, 2)
print("\n=== Manual calculation for pixel (5, 2) ===\n")

x_out, y_out = 5, 2

# Our kernel's logic
half_w = width * 0.5
half_h = height * 0.5

x_centered = x_out - half_w + 0.5
y_centered = y_out - half_h + 0.5

print(f"half_w = {half_w}, half_h = {half_h}")
print(f"x_centered = {x_out} - {half_w} + 0.5 = {x_centered}")
print(f"y_centered = {y_out} - half_h + 0.5 = {y_centered}")
print()

# Matrix elements
a, b, c_tx = 0.0, 1.0, 0.0
d, e, f_ty = -1.0, 0.0, 0.0

x_norm = (a * x_centered + b * y_centered + c_tx) / half_w
y_norm = (d * x_centered + e * y_centered + f_ty) / half_h

print(f"x_norm = ({a} * {x_centered} + {b} * {y_centered} + {c_tx}) / {half_w} = {x_norm}")
print(f"y_norm = ({d} * {x_centered} + {e} * {y_centered} + {f_ty}) / {half_h} = {y_norm}")
print()

# Compare with PyTorch's grid
pt_gx = pytorch_grid[0, y_out, x_out, 0].item()
pt_gy = pytorch_grid[0, y_out, x_out, 1].item()

print(f"Our calculation: ({x_norm:.10f}, {y_norm:.10f})")
print(f"PyTorch grid: ({pt_gx:.10f}, {pt_gy:.10f})")
print(f"Difference: ({abs(x_norm - pt_gx):.10e}, {abs(y_norm - pt_gy):.10e})")
