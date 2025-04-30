#pragma once

#define USE_CUDA
#ifdef USE_CUDA

#include "math.cuh"

namespace torchaudio {
namespace rnnt {

template <int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
__global__ void ReduceMax2D(
    int N,
    int dim,
    const DTYPE* inputs, // [N, dim]
    CAST_DTYPE* outputs)
{
  __shared__ CAST_DTYPE shared[NUM_THREADS];

#if 0
  for (int idx = blockIdx.x; idx < N; idx += blockDim.x)
  {
    if (idx >= N)
        return;

    // each thread reduces one matrix row
    long long int offset = idx;
    offset = offset * dim;
    CAST_DTYPE val = inputs[offset]; // default = inputs(n, 0)
    for (int d = threadIdx.x; d < dim; d += NUM_THREADS)
    {
        CAST_DTYPE next = inputs[offset + d];
        if (next > val)
        {
        val = next;
        }
    }


    shared[threadIdx.x] = val;
    __syncthreads();

    for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1)
    {
        if (threadIdx.x < stride && threadIdx.x + stride < dim)
        {
        if (shared[threadIdx.x + stride] > shared[threadIdx.x])
        {
            shared[threadIdx.x] = shared[threadIdx.x + stride];
            val = shared[threadIdx.x];
        }
        }
        __syncthreads();
    }

    CAST_DTYPE shf;
    for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1)
    {
        shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
        if (threadIdx.x < stride && threadIdx.x + stride < dim)
        {
            if (shf > val)
            {
                val = shf;
            }
        }
    }

    if (threadIdx.x == 0)
    {
        outputs[idx] = val;
    }
    __syncthreads();

  }

#else
  // each thread reduces one matrix row
  long long int offset = blockIdx.x;
  offset = offset * dim;
  CAST_DTYPE val = inputs[offset]; // default = inputs(n, 0)
  for (int d = threadIdx.x; d < dim; d += NUM_THREADS)
  {
    CAST_DTYPE next = inputs[offset + d];
    if (next > val)
    {
      val = next;
    }
  }


  shared[threadIdx.x] = val;
  __syncthreads();

  for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1)
  {
    if (threadIdx.x < stride && threadIdx.x + stride < dim)
    {
      if (shared[threadIdx.x + stride] > shared[threadIdx.x])
      {
        shared[threadIdx.x] = shared[threadIdx.x + stride];
        val = shared[threadIdx.x];
      }
    }
    __syncthreads();
  }

  CAST_DTYPE shf;
  for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1)
  {
    shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
    if (threadIdx.x < stride && threadIdx.x + stride < dim)
    {
      if (shf > val)
      {
        val = shf;
      }
    }
  }

  if (threadIdx.x == 0)
  {
    outputs[blockIdx.x] = val;
  }
#endif
}

template <int NUM_THREADS, typename DTYPE, typename CAST_DTYPE>
__global__ void ReduceLogSumExpGivenMax2D(
    int N,
    int dim,
    const DTYPE* inputs, // [N, dim]
    CAST_DTYPE* outputs)
{ // in: max -> out: logsum

  __shared__ CAST_DTYPE shared[NUM_THREADS];

#if 0
  for (int idx = blockIdx.x; idx < N; idx += blockDim.x)
  {
    if (idx >= N)
        return;
    CAST_DTYPE max = outputs[idx];
    CAST_DTYPE val = 0;

    long long int offset = idx;
    offset = offset * dim;
    for (int d = threadIdx.x; d < dim; d += NUM_THREADS)
    {
        val = val + std::exp(CAST_DTYPE(inputs[offset + d]) - max);
    }

    shared[threadIdx.x] = val;
    __syncthreads();

    for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1)
    {
        if (threadIdx.x < stride && threadIdx.x + stride < dim)
        {
        val = shared[threadIdx.x] + shared[threadIdx.x + stride];
        shared[threadIdx.x] = val;
        }
        __syncthreads();
    }

    CAST_DTYPE shf;
    for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1)
    {
        shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
        if (threadIdx.x < stride && threadIdx.x + stride < dim)
        {
            val = val + shf;
        }
    }

    if (threadIdx.x == 0)
    {
        outputs[idx] = max + std::log(val);
    }

  }
#else
  CAST_DTYPE max = outputs[blockIdx.x];
  CAST_DTYPE val = 0;

  long long int offset = blockIdx.x;
  offset = offset * dim;
  for (int d = threadIdx.x; d < dim; d += NUM_THREADS)
  {
    val = val + std::exp(CAST_DTYPE(inputs[offset + d]) - max);
  }

  shared[threadIdx.x] = val;
  __syncthreads();

  for (int stride = (NUM_THREADS >> 1); stride >= WARP_SIZE; stride >>= 1)
  {
    if (threadIdx.x < stride && threadIdx.x + stride < dim)
    {
      val = shared[threadIdx.x] + shared[threadIdx.x + stride];
      shared[threadIdx.x] = val;
    }
    __syncthreads();
  }

  CAST_DTYPE shf;
  for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1)
  {
    shf = __shfl_down_sync(0xFFFFFFFF, val, stride);
    if (threadIdx.x < stride && threadIdx.x + stride < dim)
    {
      val = val + shf;
    }
  }

  if (threadIdx.x == 0)
  {
    outputs[blockIdx.x] = max + std::log(val);
  }
#endif
}

} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
