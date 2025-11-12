import torch
import time
import os
import numpy as np

def hash52(p_in):
    p = p_in & 0xFFFFFFFF
    p ^= (p >> 15) & 0xFFFFFFFF
    p = (p - (p << 17)) & 0xFFFFFFFF
    p = (p + (p << 7) + (p << 4)) & 0xFFFFFFFF
    p ^= (p >> 5) & 0xFFFFFFFF
    p = (p + (p << 16)) & 0xFFFFFFFF
    p ^= (p >> 7) & 0xFFFFFFFF
    p ^= (p >> 3) & 0xFFFFFFFF
    p ^= (p << 6) & 0xFFFFFFFF
    p ^= (p >> 17) & 0xFFFFFFFF
    return p

def select_partition(seed, x, y, z, parts, num_texels):
    """A Python translation of the select_partition function from Listing 6."""
    
    if num_texels < 31:
        x <<= 1
        y <<= 1
        z <<= 1

    seed += (parts - 1) * 1024
    
    rnum = hash52(seed)

    seed1 = rnum & 0xF
    seed2 = (rnum >> 4) & 0xF
    seed3 = (rnum >> 8) & 0xF
    seed4 = (rnum >> 12) & 0xF
    seed5 = (rnum >> 16) & 0xF
    seed6 = (rnum >> 20) & 0xF
    seed7 = (rnum >> 24) & 0xF
    seed8 = (rnum >> 28) & 0xF
    seed9 = (rnum >> 18) & 0xF
    seed10 = (rnum >> 22) & 0xF
    seed11 = (rnum >> 26) & 0xF
    seed12 = ((rnum >> 30) | (rnum << 2)) & 0xF

    seed1 *= seed1
    seed2 *= seed2
    seed3 *= seed3
    seed4 *= seed4
    seed5 *= seed5
    seed6 *= seed6
    seed7 *= seed7
    seed8 *= seed8
    seed9 *= seed9
    seed10 *= seed10
    seed11 *= seed11
    seed12 *= seed12

    if (seed & 1):
        sh1 = 4 if (seed & 2) else 5
        sh2 = 6 if (parts == 3) else 5
    else:
        sh1 = 6 if (parts == 3) else 5
        sh2 = 4 if (seed & 2) else 5

    sh3 = sh1 if (seed & 0x10) else sh2

    seed1 >>= sh1
    seed2 >>= sh2
    seed3 >>= sh1
    seed4 >>= sh2
    seed5 >>= sh1
    seed6 >>= sh2
    seed7 >>= sh1
    seed8 >>= sh2
    seed9 >>= sh3
    seed10 >>= sh3
    seed11 >>= sh3
    seed12 >>= sh3

    a = (seed1 * x + seed2 * y + seed11 * z + (rnum >> 14)) & 0x3F
    b = (seed3 * x + seed4 * y + seed12 * z + (rnum >> 10)) & 0x3F
    c = (seed5 * x + seed6 * y + seed9 * z + (rnum >> 6)) & 0x3F
    d = (seed7 * x + seed8 * y + seed10 * z + (rnum >> 2)) & 0x3F

    if parts < 4: d = 0
    if parts < 3: c = 0

    if a >= b and a >= c and a >= d: return 0
    elif b >= c and b >= d: return 1
    elif c >= d: return 2
    else: return 3


def generate_valid_2p_maps():
    print("Generating and filtering 1024 2-partition ASTC maps...")
    BLOCK_WIDTH, BLOCK_HEIGHT = 4, 4
    NUM_TEXELS = BLOCK_WIDTH * BLOCK_HEIGHT
    PARTS = 2
    
    seen_normalized_maps = set()
    valid_maps_list = []
    valid_seeds_list = []

    for seed in range(1024):
        partition_map = [
            select_partition(seed, x, y, 0, PARTS, NUM_TEXELS)
            for y in range(BLOCK_HEIGHT)
            for x in range(BLOCK_WIDTH)
        ]
        valid_maps_list.append(partition_map)
        valid_seeds_list.append(seed)

    print(f"Found {len(valid_maps_list)} unique, valid 2-partition maps (Expected 437).")
    
    return torch.tensor(valid_maps_list, dtype=torch.float32), \
           torch.tensor(valid_seeds_list, dtype=torch.int16) # Use int16 for seeds

# --- Part 3: LUT Generation ---

def create_2p_ideal_to_seed_lut(valid_maps_tensor, valid_seeds_tensor, device):
    """
    Generates the 2^16 LUT by finding the closest valid map for every
    possible "ideal" map.
    
    'ideal_map' (0-65535) -> 'closest_valid_seed' (0-1023)
    """
    print(f"Generating 2^16 ({2**16}) entry 'ideal_map_to_seed' LUT...")
    N_VALID_MAPS = valid_maps_tensor.shape[0]
    print("   ... creating ideal map tensor [65536, 16]")
    ideal_maps_int = torch.arange(2**16, device=device, dtype=torch.int32)
    bit_indices = torch.arange(16, device=device, dtype=torch.int32)
    ideal_maps_binary = ((ideal_maps_int.unsqueeze(1) >> bit_indices) & 1).float()

    print(f"   ... calculating Hamming distances vs {N_VALID_MAPS} valid maps")
    valid_maps_tensor = valid_maps_tensor.to(device)
    distances = torch.cdist(ideal_maps_binary, valid_maps_tensor, p=1)
    
    print("   ... finding minimum distances (argmin)")
    closest_valid_map_indices = torch.argmin(distances, dim=1)
    
    valid_seeds_tensor = valid_seeds_tensor.to(device)
    lut = valid_seeds_tensor[closest_valid_map_indices]
    
    print("Ideal-to-Seed LUT generation complete.")
    return lut.cpu()

def generate_seed_to_mask_lut():
    print("Generating 1024-entry 'seed_to_mask' LUT...")
    BLOCK_WIDTH, BLOCK_HEIGHT = 4, 4
    NUM_TEXELS = BLOCK_WIDTH * BLOCK_HEIGHT
    PARTS = 2
    
    lut = torch.zeros(1024, dtype=torch.uint16)
    
    for seed in range(1024):
        partition_map = [
            select_partition(seed, x, y, 0, PARTS, NUM_TEXELS)
            for y in range(BLOCK_HEIGHT)
            for x in range(BLOCK_WIDTH)
        ]
        mask_int = 0
        for i in range(len(partition_map)):
            if partition_map[i] == 1:
                mask_int |= (1 << i)
                
        lut[seed] = mask_int
        
    print("Seed-to-Mask LUT generation complete.")
    return lut

if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    valid_maps_tensor, valid_seeds_tensor = generate_valid_2p_maps()
    
    # 65536-entry (ideal map -> seed) LUT
    lut_ideal_to_seed = create_2p_ideal_to_seed_lut(valid_maps_tensor, valid_seeds_tensor, device)
    # 1024-entry (seed -> mask) LUT
    lut_seed_to_mask = generate_seed_to_mask_lut()

    file1 = "astc_2p_4x4_lut.bin"
    with open(file1, "wb") as f:
        # Save as int16 as requested
        f.write(lut_ideal_to_seed.numpy().astype(np.int16).tobytes())

    print(f"\nSuccessfully generated and saved LUT 1 to '{file1}'")
    print(f"   -> Total entries: {len(lut_ideal_to_seed)}")
    print(f"   -> Data type: int16 (2 bytes per entry)")
    print(f"   -> Total size: {os.path.getsize(file1)} bytes (128 KiB)")

    file2 = "astc_2p_seed_to_mask_lut.bin"
    with open(file2, "wb") as f:
        f.write(lut_seed_to_mask.numpy().astype(np.int16).tobytes())
        
    print(f"\nSuccessfully generated and saved LUT 2 to '{file2}'")
    print(f"   -> Total entries: {len(lut_seed_to_mask)}")
    print(f"   -> Data type: int16 (2 bytes per entry)")
    print(f"   -> Total size: {os.path.getsize(file2)} bytes (2 KiB)")

    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
    
    print("\nVerifying LUT chain...")
    # Vertical split: 0011 0011 0011 0011
    split_int = 0b1100110011001100 
    
    closest_seed = lut_ideal_to_seed[split_int].item()
    print(f"Ideal vertical split (int {split_int}) maps to ASTC seed: {closest_seed}")
    
    mask_from_seed = lut_seed_to_mask[closest_seed].item()
    print(f"Seed {closest_seed} maps to 16-bit mask: {mask_from_seed:016b}")