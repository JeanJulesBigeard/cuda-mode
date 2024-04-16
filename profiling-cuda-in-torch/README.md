### Profiling Torch square:

#### Pytorch profiler:

**NB:** CUDA IS ASYNC so can't use python time module

It is possible to time a torch function using `torch.cuda.Event` as:

```
def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time modulCe
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```

Profiling three different square functions:

- torch.square
```
# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/956dbaf3-9574-47a8-a7a9-52eebd482bae)

- a*a

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/e0b2d87f-5945-4388-bce3-bd5ce7a52cee)

- a**2

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/98d421f0-682a-4521-8a21-2170fa9ca013)

NB: It allows us to see which kernel is called in the function.

On other possibility is to dump a json file and acces it in the browser.

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/6cad3bb0-75f0-4773-9e29-62d7f59d4db5)
![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/34c5fbf7-1908-4cef-b2ce-e09c2b151c8d)

What we can see: 

- Aten::square is a call to aten:pow
- A cuda kernel gets launched called native_vectorized_elementwise_kernel<4, ..> (4 is the number of blocks)

How to:
```
# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(str(prof.step_num) + ".json")
```

```
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
        
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
    ) as p:
        for iter in range(10):
            torch.square(torch.randn(10000, 10000).cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()
```


#### Custom cpp extensions:

It is possible to run cpp using torch.utils.cpp_extension as here:

NB: Build files will be automatically generated

```
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True,
    build_directory='./tmp'
)

print(my_module.hello_world())
```

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/a0247009-ae0f-4568-bbca-8a071148ae9e)

It is also possible to do so with a CUDA kernel:

```
# Define the CUDA kernel and C++ wrapper
cuda_source = '''
// CUDA kernel multiplying matrix element by itself
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

// cpp wrapper turn pytorch tensor into the correct format
torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
    }
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"
```

```
# Load the CUDA kernel as a PyTorch extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))
```

![image](https://github.com/JeanJulesBigeard/cuda-mode/assets/48935007/6355cfaf-9658-490e-86ef-571970165961)

### Triton - DSL:

Will generate a ptx (CUDA assembly) code for us.

Integration in really easy since we just need to call the kernel as a classic Python function.

Define the kernel
NB: loading row by row into the SRAM and multiply it by itself (ie we operate between rows instead of threads)
```
@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)
```

Then calling it
```
def square(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    square_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```
