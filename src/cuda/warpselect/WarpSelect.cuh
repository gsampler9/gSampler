#ifndef GS_CUDA_WARPSELECT_WARPSELECT_H_
#define GS_CUDA_WARPSELECT_WARPSELECT_H_
#include "MergeNetworkWarp.cuh"
#include "Pair.cuh"
#include "Reductions.cuh"
#include "Utils.cuh"

// From
// https://github.com/facebookresearch/faiss/blob/151e3d7be54aec844b6328dc3e7dd0b83fcfa5bc/faiss/gpu/utils/WarpSelectKernel.cuh
//
// per-warp WarpSelect
//

// `Dir` true, produce largest values.
// `Dir` false, produce smallest values.
namespace gs {
namespace impl {
namespace warpselect {

template <typename K, typename V, bool Dir, typename Comp, int NumWarpQ,
          int NumThreadQ, int ThreadsPerBlock>
struct WarpSelect {
  static constexpr int kNumWarpQRegisters = NumWarpQ / kWarpSize;

  __device__ inline WarpSelect(K initKVal, V initVVal, int k)
      : initK(initKVal),
        initV(initVVal),
        numVals(0),
        warpKTop(initKVal),
        kLane((k - 1) % kWarpSize) {
    static_assert(isPowerOf2(ThreadsPerBlock), "threads must be a power-of-2");
    static_assert(isPowerOf2(NumWarpQ), "warp queue must be power-of-2");

    // Fill the per-thread queue keys with the default value
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      threadK[i] = initK;
      threadV[i] = initV;
    }

    // Fill the warp queue with the default value
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warpK[i] = initK;
      warpV[i] = initV;
    }
  }

  __device__ inline void reset() {
    numVals = 0;
    warpKTop = initK;

// Fill the per-thread queue keys with the default value
#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      threadK[i] = initK;
      threadV[i] = initV;
    }

// Fill the warp queue with the default value
#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      warpK[i] = initK;
      warpV[i] = initV;
    }
  }

  __device__ inline void addThreadQ(K k, V v) {
    if (Dir ? Comp::gt(k, warpKTop) : Comp::lt(k, warpKTop)) {
      // Rotate right
#pragma unroll
      for (int i = NumThreadQ - 1; i > 0; --i) {
        threadK[i] = threadK[i - 1];
        threadV[i] = threadV[i - 1];
      }

      threadK[0] = k;
      threadV[0] = v;
      ++numVals;
    }
  }

  __device__ inline void checkThreadQ() {
    bool needSort = (numVals == NumThreadQ);

    needSort = __any_sync(0xffffffff, needSort);

    if (!needSort) {
      // no lanes have triggered a sort
      return;
    }

    mergeWarpQ();

    // Any top-k elements have been merged into the warp queue; we're
    // free to reset the thread queues
    numVals = 0;

#pragma unroll
    for (int i = 0; i < NumThreadQ; ++i) {
      threadK[i] = initK;
      threadV[i] = initV;
    }

    // We have to beat at least this element
    warpKTop = shfl(warpK[kNumWarpQRegisters - 1], kLane);
  }

  /// This function handles sorting and merging together the
  /// per-thread queues with the warp-wide queue, creating a sorted
  /// list across both
  __device__ inline void mergeWarpQ() {
    // Sort all of the per-thread queues
    warpSortAnyRegisters<K, V, NumThreadQ, !Dir, Comp>(threadK, threadV);

    // The warp queue is already sorted, and now that we've sorted the
    // per-thread queue, merge both sorted lists together, producing
    // one sorted list
    warpMergeAnyRegisters<K, V, kNumWarpQRegisters, NumThreadQ, !Dir, Comp,
                          false>(warpK, warpV, threadK, threadV);
  }

  /// WARNING: all threads in a warp must participate in this.
  /// Otherwise, you must call the constituent parts separately.
  __device__ inline void add(K k, V v) {
    addThreadQ(k, v);
    checkThreadQ();
  }

  __device__ inline void reduce() {
    // Have all warps dump and merge their queues; this will produce
    // the final per-warp results
    mergeWarpQ();
  }

  /// Dump final k selected values for this warp out
  __device__ inline void writeOut(K *outK, V *outV, int k) {
    int laneId = _getLaneId();

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      int idx = i * kWarpSize + laneId;

      if (idx < k) {
        outK[idx] = warpK[i];
        outV[idx] = warpV[i];
      }
    }
  }

  __device__ inline void writeOutV(V *outV, int k) {
    int laneId = _getLaneId();

#pragma unroll
    for (int i = 0; i < kNumWarpQRegisters; ++i) {
      int idx = i * kWarpSize + laneId;

      if (idx < k) {
        outV[idx] = warpV[i];
      }
    }
  }

  // Default element key
  const K initK;

  // Default element value
  const V initV;

  // Number of valid elements in our thread queue
  int numVals;

  // The k-th highest (Dir) or lowest (!Dir) element
  K warpKTop;

  // Thread queue values
  K threadK[NumThreadQ];
  V threadV[NumThreadQ];

  // warpK[0] is highest (Dir) or lowest (!Dir)
  K warpK[kNumWarpQRegisters];
  V warpV[kNumWarpQRegisters];

  // This is what lane we should load an approximation (>=k) to the
  // kth element from the last register in the warp queue (i.e.,
  // warpK[kNumWarpQRegisters - 1]).
  int kLane;
};

/// Specialization for k == 1 (NumWarpQ == 1)
template <typename K, typename V, bool Dir, typename Comp, int NumThreadQ,
          int ThreadsPerBlock>
struct WarpSelect<K, V, Dir, Comp, 1, NumThreadQ, ThreadsPerBlock> {
  static constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

  __device__ inline WarpSelect(K initK, V initV, int k)
      : threadK(initK), threadV(initV), _initK(initK), _initV(initV) {}

  __device__ inline void addThreadQ(K k, V v) {
    bool swap = Dir ? Comp::gt(k, threadK) : Comp::lt(k, threadK);
    threadK = swap ? k : threadK;
    threadV = swap ? v : threadV;
  }

  __device__ inline void checkThreadQ() {
    // We don't need to do anything here, since the warp doesn't
    // cooperate until the end
  }

  __device__ inline void add(K k, V v) { addThreadQ(k, v); }

  __device__ inline void reset() {
    threadK = _initK;
    threadV = _initV;
  }

  __device__ inline void reduce() {
    // Reduce within the warp
    Pair<K, V> pair(threadK, threadV);

    if (Dir) {
      pair =
          warpReduceAll<Pair<K, V>, Max<Pair<K, V>>>(pair, Max<Pair<K, V>>());
    } else {
      pair =
          warpReduceAll<Pair<K, V>, Min<Pair<K, V>>>(pair, Min<Pair<K, V>>());
    }

    threadK = pair.k;
    threadV = pair.v;
  }

  /// Dump final k selected values for this warp out
  __device__ inline void writeOut(K *outK, V *outV, int k) {
    if (_getLaneId() == 0) {
      *outK = threadK;
      *outV = threadV;
    }
  }

  __device__ inline void writeOutV(V *outV, int k) {
    if (_getLaneId() == 0) {
      *outV = threadV;
    }
  }

  // threadK is lowest (Dir) or highest (!Dir)
  K threadK;
  V threadV;

  // Default element key
  const K _initK;

  // Default element value
  const V _initV;
};
}  // namespace warpselect
}  // namespace impl
}  // namespace gs
#endif