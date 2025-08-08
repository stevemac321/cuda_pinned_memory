#pragma once
// Standard Library
#include <algorithm>
#include <array>
#include <cctype>           // std::isspace
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>           // std::unique_ptr
#include <sstream>
#include <string>
#include <vector>
#include <complex>
#include <cstddef>
#include <algorithm>
#include <cmath>

// Windows API
#include <windows.h>
#include <cufft.h>
#include "mkl.h"
#include <oleidl.h>

extern std::ofstream out; // global perf log for the entire program, not just runs.


struct SignalReport {
  int chunk_index = 0;

  struct FrequencyResponse {
    std::vector<std::vector<float>> matrix;
    std::vector<float> dominant_frequency;
    std::vector<float> spectral_centroid;
    std::vector<float> spectral_spread;
    std::vector<float> power;
  } freq_response;

  std::vector<std::complex<float>> chunk_spectrum;
  std::vector<float> magnitudes;
  std::vector<std::vector<float>> spectrum_matrix;


  // Set chunk index
  void set_chunk_index(int idx) { chunk_index = idx; }

  // Compute and store dominant frequency
  void compute_dominant_frequency(const std::vector<float> &magnitudes) {
    auto max_it = std::max_element(magnitudes.begin(), magnitudes.end());
    float value = std::distance(magnitudes.begin(), max_it);
    freq_response.dominant_frequency.push_back(value);
  }

  // Compute and store spectral centroid
  void compute_spectral_centroid(const std::vector<float> &magnitudes) {
    float sum = 0.0f, weighted_sum = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
      weighted_sum += i * magnitudes[i];
      sum += magnitudes[i];
    }
    float value = (sum > 0.0f) ? weighted_sum / sum : 0.0f;
    freq_response.spectral_centroid.push_back(value);
  }

  // Compute and store spectral spread
  void compute_spectral_spread(const std::vector<float> &magnitudes) {
    float centroid =
        freq_response.spectral_centroid.back(); // use last computed centroid
    float sum = 0.0f, variance = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
      float diff = i - centroid;
      variance += diff * diff * magnitudes[i];
      sum += magnitudes[i];
    }
    float value = (sum > 0.0f) ? std::sqrt(variance / sum) : 0.0f;
    freq_response.spectral_spread.push_back(value);
  }

  // Compute and store power
  void compute_power(const std::vector<float> &magnitudes) {
    float total = 0.0f;
    for (float m : magnitudes)
      total += m * m;
    freq_response.power.push_back(total);
  }

  // Append magnitudes to matrix
  void append_spectrum(const std::vector<float> &magnitudes) {
    freq_response.matrix.push_back(magnitudes);
  }

  // Dump all data to text file
  void dump_to_text(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
      std::cerr << "Failed to open file: " << filename << "\n";
      return;
    }

    out << "Spectrum Report (" << freq_response.matrix.size()
        << " chunks):\n\n";

    for (size_t i = 0; i < freq_response.matrix.size(); ++i) {
      out << "Chunk Index: " << i << "\n";
      out << "Dominant Frequency: " << freq_response.dominant_frequency[i]
          << "\n";
      out << "Spectral Centroid: " << freq_response.spectral_centroid[i]
          << "\n";
      out << "Spectral Spread: " << freq_response.spectral_spread[i] << "\n";
      out << "Power per Batch: " << freq_response.power[i] << "\n";
      out << "Spectrum:";
      for (float val : freq_response.matrix[i]) {
        out << " " << val;
      }
      out << "\n\n";
    }

    out.close();
  }
  void accumulate_spectrum(const float *fft_output,
                                         size_t chunk_size,
                                         size_t chunk_index) {
    // Convert interleaved FFT output to complex<float>
    chunk_spectrum.clear();
    chunk_spectrum.reserve(chunk_size);
    for (size_t k = 0; k < chunk_size; ++k) {
      float real = fft_output[k * 2 + 0];
      float imag = fft_output[k * 2 + 1];
      chunk_spectrum.emplace_back(real, imag);
    }

    // Compute magnitudes
    magnitudes.clear();
    magnitudes.reserve(chunk_size);
    for (const auto &c : chunk_spectrum)
      magnitudes.push_back(std::abs(c));

    // Accumulate metrics
    set_chunk_index(chunk_index);
    compute_dominant_frequency(magnitudes);
    compute_spectral_centroid(magnitudes);
    compute_spectral_spread(magnitudes);
    compute_power(magnitudes);
    append_spectrum(magnitudes);
  }
};

///////////////////////////////////////////////////////////
struct FileMapping {
  HANDLE hFile = nullptr;
  HANDLE hMapping = nullptr;
  void *mapped_ptr = nullptr;
  size_t size = 0;
  // Default constructor
  FileMapping() = default;



  // Destructor
  ~FileMapping() {
    if (mapped_ptr)
      UnmapViewOfFile(mapped_ptr);
    if (hMapping)
      CloseHandle(hMapping);
    if (hFile)
      CloseHandle(hFile);
  }

  // Move constructor
  FileMapping(FileMapping &&other) noexcept
      : hFile(other.hFile), hMapping(other.hMapping),
        mapped_ptr(other.mapped_ptr), size(other.size) {
    other.hFile = nullptr;
    other.hMapping = nullptr;
    other.mapped_ptr = nullptr;
    other.size = 0;
  }

  // Move assignment
  FileMapping &operator=(FileMapping &&other) noexcept {
    if (this != &other) {
      // Clean up current resources
      if (mapped_ptr) {

        UnmapViewOfFile(mapped_ptr);
        cudaFreeHost(mapped_ptr);
      }
      if (hMapping)
        CloseHandle(hMapping);
      if (hFile)
        CloseHandle(hFile);

      // Transfer ownership
      hFile = other.hFile;
      hMapping = other.hMapping;
      mapped_ptr = other.mapped_ptr;
      size = other.size;

      // Null out other's handles
      other.hFile = nullptr;
      other.hMapping = nullptr;
      other.mapped_ptr = nullptr;
      other.size = 0;
    }
    return *this;
  }

  // Disable copy semantics
  FileMapping(const FileMapping &) = delete;
  FileMapping &operator=(const FileMapping &) = delete;
};

FileMapping OpenMappedFile(const std::wstring &filepath);
void cuda_fft(const char *mapdir, const size_t chunk_size = 8192);

void log_fft(const char *label, size_t rows, size_t chunk_size,
             std::chrono::nanoseconds elapsed);


