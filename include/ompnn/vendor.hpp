// SPDX-License-Identifier: LGPL-3.0-only
// ompnn - vendor runtime/BLAS spelling for the GPU path.  The code is written
// against the CUDA/cuBLAS names; on AMD the same names map to HIP/hipBLAS.
#pragma once

#if defined(OMPNN_TARGET_NVIDIA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#elif defined(OMPNN_TARGET_AMD)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaGetLastError hipGetLastError
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetPointerMode hipblasSetPointerMode
#define CUBLAS_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define cublasOperation_t hipblasOperation_t
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_N HIPBLAS_OP_N
#define cublasSgemm hipblasSgemm
#define cublasDgemm hipblasDgemm
#define cublasSgemv hipblasSgemv
#define cublasDgemv hipblasDgemv
#define cublasSasum hipblasSasum
#define cublasDasum hipblasDasum
#define cublasSnrm2 hipblasSnrm2
#define cublasDnrm2 hipblasDnrm2
#endif
