import wgpu
import wgpu.utils
import wgpu.enums
import numpy as np
import asyncio
import os
import sys
from PIL import Image


def prepare_input_data(filename="input.png"):
    with Image.open(filename) as img:
        original_width, original_height = img.size

        img_rgba = img.convert('RGBA')
        img_data = np.array(img_rgba, dtype=np.float32) / 255.0

        padded_width = (original_width + 3) // 4 * 4
        padded_height = (original_height + 3) // 4 * 4
        total_blocks = (padded_width // 4) * (padded_height // 4)

        padded_array = np.zeros((padded_height, padded_width, 4), dtype=np.float32)
        padded_array[:original_height, :original_width, :] = img_data

        reshaped = padded_array.reshape(padded_height // 4, 4, padded_width // 4, 4, 4)
        transposed = reshaped.transpose(0, 2, 1, 3, 4)
        final_reshape = transposed.reshape(total_blocks, 16, 4)
        tiled_data = final_reshape.reshape(total_blocks * 16, 4).astype(np.float32)

        return tiled_data, total_blocks, original_width, original_height


async def run_astc_compute(input_data, total_blocks):
    """
    The main WGPU function, now returning both compressed and decoded data.
    """
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("Could not get a WGPU adapter.")
    
    features = adapter.features
    print(f"Adapter features: {features}")

    if "timestamp-query" not in features:
        raise RuntimeError("Timestamp queries not supported on this adapter. Cannot perform precise timing.")
    device = adapter.request_device_sync(required_features=["timestamp-query", "subgroup", 'subgroup-barrier'])

    print(f"Got device: {adapter.summary}")

    with open("astc_compress.spv", "rb") as f:
        shader_code = f.read()
    shader_module = device.create_shader_module(code=shader_code)

    NUM_WORKGROUPS = (total_blocks + 3) // 4
    total_blocks_padded_for_shader = NUM_WORKGROUPS * 4

    cparams_data = np.array([total_blocks], dtype=np.uint32)
    b0_uniform = device.create_buffer_with_data(data=cparams_data, usage=wgpu.BufferUsage.UNIFORM)
    input_pixel_buffer_size = input_data.nbytes
    b1_storage_in = device.create_buffer_with_data(data=input_data, usage=wgpu.BufferUsage.STORAGE)
    compressed_buffer_size = total_blocks_padded_for_shader * 16  # Each uvec4 is 16 bytes
    b2_storage_out_compressed = device.create_buffer(size=compressed_buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    b3_storage_out_decoded = device.create_buffer(size=input_pixel_buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    b4_readback_compressed = device.create_buffer(size=compressed_buffer_size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
    b5_readback_decoded = device.create_buffer(size=input_pixel_buffer_size, usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        ]
    )
    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": b0_uniform, "offset": 0, "size": b0_uniform.size}},
            {"binding": 1, "resource": {"buffer": b1_storage_in, "offset": 0, "size": b1_storage_in.size}},
            {"binding": 2, "resource": {"buffer": b2_storage_out_compressed, "offset": 0, "size": b2_storage_out_compressed.size}},
            {"binding": 3, "resource": {"buffer": b3_storage_out_decoded, "offset": 0, "size": b3_storage_out_decoded.size}},
        ],
    )

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )

    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group, [], 0, 99999)
    compute_pass.dispatch_workgroups(NUM_WORKGROUPS)
    compute_pass.end()

    command_encoder.copy_buffer_to_buffer(b2_storage_out_compressed, 0, b4_readback_compressed, 0, compressed_buffer_size)
    command_encoder.copy_buffer_to_buffer(b3_storage_out_decoded, 0, b5_readback_decoded, 0, input_pixel_buffer_size)

    queue = device.queue
    queue.submit([command_encoder.finish()])
    print("Waiting for GPU...")

    import time
    start_time = time.perf_counter()

    await b4_readback_compressed.map_async(wgpu.MapMode.READ, 0, compressed_buffer_size)
    await b5_readback_decoded.map_async(wgpu.MapMode.READ, 0, input_pixel_buffer_size)

    end_time = time.perf_counter()
    gpu_duration_ms = (end_time - start_time) * 1000
    print(f"GPU execution and data readback took: {gpu_duration_ms:.2f} ms")

    mapped_compressed = b4_readback_compressed.read_mapped(0, compressed_buffer_size)
    compressed_data = np.frombuffer(mapped_compressed, dtype=np.uint32).copy()
    b4_readback_compressed.unmap()

    mapped_decoded = b5_readback_decoded.read_mapped(0, input_pixel_buffer_size)
    decoded_pixel_data = np.frombuffer(mapped_decoded, dtype=np.float32).copy()
    b5_readback_decoded.unmap()

    reshaped_compressed = compressed_data.reshape((total_blocks_padded_for_shader, 4))
    final_compressed_data = reshaped_compressed[:total_blocks]

    return final_compressed_data, decoded_pixel_data


def save_astc_file(filename, results, width, height):
    """
    Writes the raw GPU data and a 16-byte ASTC header to a file.
    """
    header = bytearray(b'\x13\xAB\xA1\x5C')  # Magic
    header.extend([4, 4, 1])               # block_dims
    header.extend(width.to_bytes(3, 'little'))
    header.extend(height.to_bytes(3, 'little'))
    header.extend((1).to_bytes(3, 'little')) # z-dim

    try:
        with open(filename, "wb") as f:
            f.write(header)
            f.write(results.tobytes())
        print(f"Successfully saved compressed data to '{filename}'")
    except Exception as e:
        print(f"Error writing ASTC file: {e}")


def save_decoded_image(filename, flat_pixel_data, original_width, original_height):
    padded_width = (original_width + 3) // 4 * 4
    padded_height = (original_height + 3) // 4 * 4
    num_blocks_x = padded_width // 4
    num_blocks_y = padded_height // 4

    # The data is currently (total_pixels, 4), which is (num_blocks_y * num_blocks_x * 16, 4)
    pixels_as_blocks = flat_pixel_data.reshape(num_blocks_y, num_blocks_x, 16, 4)
    tiles = pixels_as_blocks.reshape(num_blocks_y, num_blocks_x, 4, 4, 4)
    unswizzled = tiles.transpose(0, 2, 1, 3, 4)
    padded_image = unswizzled.reshape(padded_height, padded_width, 4)
    cropped_image = padded_image[:original_height, :original_width, :]
    image_uint8 = (np.clip(cropped_image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image_uint8, 'RGBA').save(filename)


def main():
    input_filename = "screenshot.jpg"
    astc_output_filename = "output.astc"
    png_output_filename = "decoded_screenshot.png"
    tiled_data, total_blocks, width, height = prepare_input_data(input_filename)
    if tiled_data is None:
        return

    compressed_results, decoded_pixels = asyncio.run(run_astc_compute(tiled_data, total_blocks))
    save_astc_file(astc_output_filename, compressed_results, width, height)
    save_decoded_image(png_output_filename, decoded_pixels, width, height)


if __name__ == "__main__":
    main()
