from numba import cuda
print(cuda.is_available())
print(cuda.detect())

device = cuda.get_current_device()

print("Max threads per block:", device.MAX_THREADS_PER_BLOCK)
print("Max block dimensions:", device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y, device.MAX_BLOCK_DIM_Z)
print("Max grid dimensions:", device.MAX_GRID_DIM_X, device.MAX_GRID_DIM_Y, device.MAX_GRID_DIM_Z)
print("Warp size:", device.WARP_SIZE)
print("Shared memory per block (bytes):", device.MAX_SHARED_MEMORY_PER_BLOCK)
