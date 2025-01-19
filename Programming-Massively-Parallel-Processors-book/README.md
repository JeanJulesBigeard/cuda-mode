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