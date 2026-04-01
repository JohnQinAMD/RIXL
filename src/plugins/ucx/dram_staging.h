/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NIXL_SRC_PLUGINS_UCX_DRAM_STAGING_H
#define NIXL_SRC_PLUGINS_UCX_DRAM_STAGING_H

#include <cstddef>
#include <cstdint>
#include <optional>

/*
 * Transparent DRAM staging for RDMA NICs that lack GPU Direct RDMA
 * (e.g. AMD Pensando ionic).
 *
 * Uses dlopen("libamdhip64.so") at runtime so there is zero link-time
 * dependency on ROCm/HIP.  On systems without HIP the manager reports
 * isAvailable() == false and all staging paths are skipped.
 */

struct StagingInfo {
    void   *gpu_addr;
    void   *host_addr;
    size_t  size;
};

/* Wire-format header prepended to rkey blob when staging is active */
struct StagingWireHeader {
    static constexpr uint32_t MAGIC = 0x53544147; /* "STAG" */
    uint32_t magic;
    uint64_t gpu_base;
    uint64_t host_base;
    uint64_t region_size;
} __attribute__((packed));

class DramStagingManager {
public:
    DramStagingManager();
    ~DramStagingManager();

    DramStagingManager(const DramStagingManager &) = delete;
    DramStagingManager &operator=(const DramStagingManager &) = delete;

    bool isAvailable() const noexcept { return available_; }

    void *allocPinned(size_t bytes);
    void  freePinned(void *ptr);

    void copyD2H(void *host_dst, const void *gpu_src, size_t bytes);
    void copyH2D(void *gpu_dst, const void *host_src, size_t bytes);
    void sync();

private:
    void *hip_lib_ = nullptr;
    bool  available_ = false;

    using hipHostMalloc_fn  = int (*)(void **, size_t, unsigned int);
    using hipHostFree_fn    = int (*)(void *);
    using hipMemcpy_fn      = int (*)(void *, const void *, size_t, int);
    using hipDeviceSync_fn  = int (*)();
    using hipSetDevice_fn   = int (*)(int);

    hipHostMalloc_fn  hipHostMalloc_  = nullptr;
    hipHostFree_fn    hipHostFree_    = nullptr;
    hipMemcpy_fn      hipMemcpy_     = nullptr;
    hipDeviceSync_fn  hipDeviceSync_  = nullptr;
    hipSetDevice_fn   hipSetDevice_   = nullptr;
};

#endif
