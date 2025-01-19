# PMPP book

### Chapter 1: Introduction

The book focuses on programming massively parallel processors to achieve high performance. The emphasis is on techniques for developing high-performance parallel code.

Parallel programming is important due to the need to increase the execution speed of applications.

The book aims to provide an intuitive and practical understanding of data management techniques to application developers.

The goal is to help the reader understand the optimizations needed to work around memory bandwidth limitations and to become familiar with their usage.

The book addresses the challenges involved in designing parallel algorithms with the same algorithmic complexity as sequential algorithms. It introduces the concept of work efficiency and the tradeoffs involved in designing parallel algorithms that achieve the same level of complexity as their sequential counterparts.

It is important to have a good conceptual understanding of parallel hardware architectures to be able to analyze the performance behavior of code. Chapter 4 is dedicated to the fundamentals of GPU architecture.

The book is organized into four parts: fundamental concepts, parallel patterns, advanced patterns, and advanced practices.


### Chapter 2: Heterogeneous Data Parallel Computing

This chapter introduces the concept of data parallelism, where the same operation is performed simultaneously on multiple pieces of data. It explains how data parallelism can be leveraged to speed up applications, for example, color-to-grayscale conversion.

The chapter presents the basic structure of a CUDA C program, distinguishing between parts executed on the host (CPU) and on the device (GPU). It explains how to declare variables for the host and device, using the "_h" and "_d" suffixes respectively.

The chapter describes the process of writing a CUDA C kernel, a function that is executed by parallel threads on the GPU. It details the distribution of data between host and device memory.

It introduces the CUDA C extensions to the C language, and describes how a kernel is launched for execution by parallel threads. It's important to note that the kernel function does not have a loop corresponding to the host program's loop.

The chapter highlights an essential subset of CUDA C extensions for writing a simple program in CUDA C.


### Chapter 3: Multidimensional Grids and Data

The chapter explains how to organize threads in multidimensional grids and blocks to process multidimensional data. It describes how threads are organized and related to resources and how data is related to threads in CUDA C.

It provides information on how the threadIdx, blockIdx, and blockDim variables are used by threads in kernel functions to identify the portion of data to be processed.

The chapter explains how to linearize multidimensional indices into a 1D offset.

### Chapter 4: Compute Architecture and Scheduling

The chapter describes the architecture of GPUs, focusing on how compute cores are organized and how threads are scheduled to execute on these cores.

It examines concepts such as transparent scalability, SIMD execution and control divergence, multithreading and latency tolerance, and occupancy, explaining their definitions and their influence on code performance.

The chapter explains the importance of maximizing the use of GPU compute cores and how queues of threads ready to be executed help tolerate the latency of instructions.

### Chapter 5: Memory Architecture and Data Locality

The chapter discusses the GPU memory architecture, introducing different memory types: global memory, constant memory, shared memory, local memory, and registers. It covers memory allocation, lifetime, visibility, accessibility, and performance of these memory types.

The chapter emphasizes the importance of memory access efficiency in achieving good performance in parallel computing.

The chapter explains how to reduce memory traffic by using tiling, a technique that involves dividing data into blocks and processing them collaboratively by the threads within the same block.

It explains how to use shared memory to efficiently share data between the threads of a block.

### Chapter 6: Performance Considerations

Memory Coalescing: This concept is crucial for improving the efficiency of global memory accesses. The chapter explains how memory accesses can be combined into a single access when threads within the same warp access adjacent memory locations. The analogy of carpooling is used to illustrate this: when multiple threads access nearby data, they can "carpool" to make a single request to the DRAM memory, reducing overall memory traffic. Coalesced memory access is essential for maximizing global memory bandwidth. The chapter also demonstrates how to reorganize data access or use shared memory to achieve coalesced access, even when the initial access pattern is unfavorable.

Hiding Memory Latency: Latency is the delay between a memory read/write request and its completion. The chapter explains how the DRAM memory architecture, with its multiple banks and channels, allows data to be transferred in parallel. Interleaving data distribution across different banks and channels in DRAM improves the utilization of data transfer bandwidth. The chapter emphasizes that maximizing thread occupancy on the SM (Streaming Multiprocessor) is crucial to tolerate long latency operations and ensure that enough memory requests are generated to fully utilize memory bandwidth.

Thread Coarsening: The chapter explores the concept of thread coarsening, where multiple parallel work units are assigned to the same thread to reduce the overhead associated with parallelism. This technique involves serializing certain parts of the work, reducing the number of threads and synchronization overhead. The chapter emphasizes that there are trade-offs between increasing the workload of a thread and reducing the overhead of parallelization. Coarsening is applied in different contexts to reduce redundant loading of input data.

Checklist of Optimizations: The chapter provides a list of common optimizations, serving as a reference for the rest of the book. This list includes optimizing thread occupancy, using coalesced accesses to global memory, caching, data privatization, and thread coarsening.

Knowing Your Computation's Bottleneck: The chapter emphasizes the need to understand the bottleneck of a computation to target optimization efforts effectively.