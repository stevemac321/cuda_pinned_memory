/*

*/
#include <cuda_runtime.h>
#include "fft_wrapper.h"
#include <filesystem> // For directory iteration
const char *cuda_report_file = "cuda_report.txt";

namespace fs = std::filesystem;

const size_t CHUNK_THRESHOLD = 16384;
const size_t REQUIRED_FILESIZE = 16777216;

// #define LOG_TELEMETRY
// #define LOG_CUDA

FileMapping OpenMappedFile(const std::wstring &filepath) {
  HANDLE hFile =
      CreateFileW(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

  if (hFile == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to open file: "
              << std::string(filepath.begin(), filepath.end()) << "\n";
    return FileMapping();
  }

  HANDLE hMapping =
      CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hMapping) {
    std::cerr << "Failed to create file mapping: "
              << std::string(filepath.begin(), filepath.end()) << "\n";
    CloseHandle(hFile);
    return FileMapping();
  }
  

  size_t filesize = static_cast<size_t>(GetFileSize(hFile, nullptr));
  if (filesize != REQUIRED_FILESIZE)  {
        std::cerr << "Unexpected file size: " << filesize << "\n";
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return FileMapping(); // default-constructed, signals failure
  
    std::cerr << "Invalid file size: "
              << std::string(filepath.begin(), filepath.end()) << "\n";
    CloseHandle(hMapping);
    CloseHandle(hFile);
    return FileMapping();
  }
  

  void *mapped = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
  if (!mapped) {
    std::cerr << "MapViewOfFile failed: "
              << std::string(filepath.begin(), filepath.end()) << "\n";
    CloseHandle(hMapping);
    CloseHandle(hFile);
    return FileMapping();
  }

  FileMapping result;
  result.hFile = hFile;
  result.hMapping = hMapping;
  result.size = filesize;
  
  float *pinned_buffer = nullptr;
  cudaError_t err = cudaMallocHost(reinterpret_cast<void **>(&pinned_buffer), 16777216);
  result.mapped_ptr = pinned_buffer;

  if (err != cudaSuccess) {
    std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << "\n";
    return FileMapping();
  }

  return result;
}

 
/////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
void cuda_fft(const char *mapdir, const size_t chunk_size) {
#ifdef LOG_TELEMETRY
  struct SignalReport report;
#endif
  std::vector<FileMapping> mapped_files;
  auto start = std::chrono::high_resolution_clock::now();

  // Allocate pinned host buffers
  float *voltage = nullptr;
  cufftComplex *fft_input = nullptr;
  cudaMallocHost(&voltage, chunk_size * sizeof(float));
  cudaMallocHost(&fft_input, chunk_size * sizeof(cufftComplex));

  // Allocate device buffer
  cufftComplex *d_data = nullptr;
  cudaMalloc(&d_data, chunk_size * sizeof(cufftComplex));

  // Create FFT plan
  cufftHandle plan;
  cufftResult plan_status = cufftPlan1d(&plan, chunk_size, CUFFT_C2C, 1);
  if (plan_status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "FFT plan creation failed: %d\n", plan_status);
    cudaFree(d_data);
    cudaFreeHost(voltage);
    cudaFreeHost(fft_input);
    return;
  }

  std::printf("Beginning CudaMemMapFFT, chunk size: %zu\n", chunk_size);
  size_t row = 0;

  for (const auto &entry : std::filesystem::directory_iterator(mapdir)) {
    if (!entry.is_regular_file())
      continue;
    const std::wstring filepath = entry.path().wstring();
    FileMapping fm = OpenMappedFile(filepath);
    if (fm.mapped_ptr)
      mapped_files.push_back(std::move(fm));
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (const auto &fm : mapped_files) {
    float *data_ptr = reinterpret_cast<float *>(fm.mapped_ptr);
    size_t usable_floats = fm.size / sizeof(float);
    size_t chunks_in_file = usable_floats / chunk_size;

    for (size_t i = 0; i < chunks_in_file; ++i, ++row) {
      std::memcpy(voltage, data_ptr + i * chunk_size,
                  chunk_size * sizeof(float));

      for (size_t j = 0; j < chunk_size; ++j) {
        fft_input[j].x = voltage[j];
        fft_input[j].y = 0.0f;
      }

      cudaMemcpyAsync(d_data, fft_input, chunk_size * sizeof(cufftComplex),
                      cudaMemcpyHostToDevice, stream);

      cufftResult result = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS) {
        std::printf("FFT failed: %d\n", result);
        continue;
      }

      cudaMemcpyAsync(fft_input, d_data, chunk_size * sizeof(cufftComplex),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);

#ifdef LOG_TELEMETRY
      report.accumulate_spectrum(fft_input, chunk_size, i);
#endif
#ifdef LOG_CUDA
      for (size_t k = 0; k < chunk_size; ++k) {
        float real = fft_input[k].x;
        float imag = fft_input[k].y;
        std::printf("Bin %4zu: % .6f + % .6fi\n", k, real, imag);
      }
#endif
    }
  }
  
  cudaStreamDestroy(stream);
  cufftDestroy(plan);
  cudaFree(d_data);
  cudaFreeHost(voltage);
  cudaFreeHost(fft_input);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  log_fft("Cuda fft", row, chunk_size, elapsed);

#ifdef LOG_TELEMETRY
  report.dump_to_text(cuda_report_file);
#endif
}

void log_fft(const char *label, size_t rows, size_t chunk_size,
             std::chrono::nanoseconds elapsed) {
  size_t total_floats = rows * chunk_size;
  double elapsed_ms = elapsed.count() / 1e6;
  double ns_per_float = static_cast<double>(elapsed.count()) / total_floats;

  // Print to stdout
  std::printf("%s FFT (%zu rows), %zu floats took %.2f ms (%.2f ns/float)\n",
              label, rows, total_floats, elapsed_ms, ns_per_float);
  // Only enable logging if none of these are defined
#if !defined(LOG_TELEMETRY) && !defined(LOG_CUDA) && !defined(LOG_MKL)

  // Append to perf.txt
  if (out) {
    out << label << "," << rows << "," << chunk_size << "," << total_floats
        << "," << elapsed_ms << "," << ns_per_float << "\n";
  }
  #endif
}