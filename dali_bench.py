import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import torch
import time
import numpy as np
import nvidia.dali.types as types
from typing import Callable, Any

# --- PARAMETERS (MATCHING YOUR SCENARIO) ---
BATCH_SIZE = 32
CHANNELS = 3
H_INIT, W_INIT = 600, 600 
crop_size = 512 
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225] 

# FIX 1: Create a single 3D image sample (H, W, C)
# This 3D shape (1024, 1024, 3) matches the 3-character layout 'HWC'.
dummy_sample = (torch.rand(H_INIT, W_INIT, CHANNELS) * 255).byte().numpy()

# --- EXTERNAL SOURCE FEEDER FUNCTION ---
# This generator function provides the data to the DALI pipeline on demand.
def external_source_feeder():
    # Since this is a fixed benchmark, we yield the same batch indefinitely.
    # FIX 2: Create the batch by yielding a list of BATCH_SIZE individual 3D image samples.
    # DALI handles the batching of these samples internally.
    batch_data = [dummy_sample] * BATCH_SIZE
    while True:
        yield batch_data

# --- OPTIMIZED DALI PIPELINE DEFINITION ---
class OptimizedAugPipe(Pipeline):
    def __init__(self, batch_size: int, crop_size: int, device_id: int = 0):
        # Initializing the pipeline
        super().__init__(batch_size=batch_size, num_threads=4, device_id=device_id, seed=42)
        
        # 1. External source: Now uses the `source` argument to link to the feeder function.
        self.input = fn.external_source(
            source=external_source_feeder, 
            device="cpu", 
            dtype=types.UINT8,
            # 'HWC' is correct because we are now yielding 3D samples.
            layout="HWC",
            name="input_data"
        ) 
        self.crop_size = crop_size
        self.flip = fn.random.coin_flip(probability=0.5)

    def define_graph(self):
        imgs = self.input
        
        # CRITICAL: Explicit CPU-to-GPU Copy
        # The copy happens after the external source, moving the data to the GPU memory space.
        imgs = fn.copy(imgs, device="gpu") 
         
        # 2. ColorJitter random parameters
        brightness = fn.random.uniform(range=[0.8, 1.2])
        contrast = fn.random.uniform(range=[0.8, 1.2])
        saturation = fn.random.uniform(range=[0.8, 1.2])

        # Apply Brightness and Contrast
        imgs = fn.brightness_contrast(imgs, brightness=brightness, contrast=contrast)
        
        # Apply Saturation
        imgs = fn.hsv(imgs, saturation=saturation)
        
        # 3. Final Fused Operation: CROP, MIRROR, NORMALIZE (CMNP)
        output = fn.crop_mirror_normalize(
            imgs,
            dtype=types.FLOAT, 
            mean=mean, 
            std=std, 
            mirror=self.flip, # Horizontal flip
            output_layout="CHW" # Output PyTorch standard layout
        )
        # Return the output under the name 'data' for DALIGenericIterator
        return output

# --- BENCHMARK FUNCTION (USING DALIGENERICITERATOR) ---
def dali_bench(pipe_class: Callable, batch_size: int, crop_size: int, warmup: int=25, reps: int=100) -> float:
    """Benchmarks the DALI pipeline using DALIGenericIterator for stable queue management."""
    
    # Instantiate and build the pipeline
    pipe = pipe_class(batch_size=batch_size, crop_size=crop_size)
    pipe.build()
    
    # DALIGenericIterator automatically manages the queue and pipeline state
    dali_iterator = DALIGenericIterator(
        pipelines=[pipe],
        output_map=['data'], 
        size=-1 # Infinite source
    )

    # 1. Warmup
    for i, _ in enumerate(dali_iterator):
        if i >= warmup:
            break
            
    # CRITICAL: Reset the iterator to ensure the pipeline is clear before timing
    dali_iterator.reset() 

    start = time.time()
    
    # 2. Timing loop
    for i, _ in enumerate(dali_iterator):
        if i >= reps:
            break
            
    torch.cuda.synchronize()
    
    avg_time = (time.time() - start) / reps
    
    # Final reset ensures the iterator is clean for future runs (if any)
    dali_iterator.reset()
        
    return avg_time

# --- EXECUTION ---
# Pass the class and parameters to the benchmark function
avg_time = dali_bench(OptimizedAugPipe, BATCH_SIZE, crop_size) 
print("-" * 50)
print(f"DALI avg time per iteration (Batch={BATCH_SIZE}): {avg_time*1000:.3f} ms")
print(f"DALI Throughput: {BATCH_SIZE / avg_time:.1f} images/second")
print("-" * 50)