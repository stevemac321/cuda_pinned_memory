#this is a continuation of the last project that compared Intel MKL with NVIDIA Cuda SDK. 
https://github.com/stevemac321/cuda_fft_batch_mmap_1g
# cuda_pinned_memory

High-throughput, pinned-memory-optimized FFT pipeline using CUDA and memory-mapped files.

This project builds on the techniques demonstrated in [`cuda_fft_batch_mmap_1g`](https://github.com/YOUR_USERNAME/cuda_fft_batch_mmap_1g), but removes all MKL dependencies and focuses exclusively on CUDA performance. It uses pinned host memory (`cudaMallocHost`) for efficient staging and transfer, and supports chunked FFT execution over memory-mapped telemetry files.

## 🔧 Features

- Fully pinned memory pipeline for host-to-device transfers
- cuFFT-based batched FFT execution
- Memory-mapped file ingestion with chunked processing
- Performance logging and optional system utilization tracking
- Tunable chunk size for latency vs throughput tradeoff

