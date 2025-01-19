# Key Points About Memory Coalescing

**Coalesced access** on GPUs means that **consecutive threads in the same warp** access **consecutive (or suitably patterned) memory addresses** in global memory. This layout allows the GPU to bundle these accesses into fewer memory transactions, significantly improving performance.

---

## Loading Matrix \( M \)

When loading elements of \( M \) into shared memory, the address pattern is:

M[row * width + (ph * TILE_WIDTH + tx)]


- **`row`** is constant for all threads in the same block row.
- **`tx`** (thread index along the x-dimension) varies from 0 to `(TILE_WIDTH - 1)`.

Because of this setup, for a given row, threads access consecutive columns in memory. This results in a **contiguous block** of floats that aligns well with GPU memory transactions, ensuring **coalesced reads** for matrix \( M \).

---

## Loading Matrix \( N \)

When loading elements of \( N \) into shared memory, the address pattern is:

N[(ph * TILE_WIDTH + ty) * width + col]


- **`col`** is computed as `colstart + c * TILE_WIDTH` and remains the same for the current unrolled portion.
- **`ty`** (thread index along the y-dimension) varies from 0 to `(TILE_WIDTH - 1)`.

Here, the threads in a warp read consecutive rows in \( N \). As a result, they access consecutive addresses in memory, making reads from \( N \) **coalesced** as well.

---

## Why It Matters

By carefully structuring our tile and thread indexing:

- **\( \mathbf{M} \)** loading is coalesced because threads in the same row read consecutive addresses.
- **\( \mathbf{N} \)** loading is coalesced because threads in the same column read consecutive addresses.

This approach reduces the number of memory transactions required, leading to higher bandwidth utilization and faster overall execution of the matrix multiplication kernel.
