#pragma once

#ifdef CPPIMPACT_CUDA_BACKEND
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CPPIMPACT_FUNCTION __host__ __device__
#else
#define CPPIMPACT_FUNCTION
#endif