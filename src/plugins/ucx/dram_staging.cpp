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

#include "dram_staging.h"
#include "common/nixl_log.h"

#include <dlfcn.h>
#include <cstring>

static constexpr int HIP_MEMCPY_DEFAULT = 4;
static constexpr unsigned int HIP_HOST_MALLOC_DEFAULT = 0;

DramStagingManager::DramStagingManager() {
    hip_lib_ = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hip_lib_) {
        NIXL_DEBUG << "DRAM staging: libamdhip64.so not found (" << dlerror()
                   << "), staging disabled";
        return;
    }

    hipHostMalloc_ =
        reinterpret_cast<hipHostMalloc_fn>(dlsym(hip_lib_, "hipHostMalloc"));
    hipHostFree_ =
        reinterpret_cast<hipHostFree_fn>(dlsym(hip_lib_, "hipHostFree"));
    hipMemcpy_ =
        reinterpret_cast<hipMemcpy_fn>(dlsym(hip_lib_, "hipMemcpy"));
    hipDeviceSync_ =
        reinterpret_cast<hipDeviceSync_fn>(dlsym(hip_lib_, "hipDeviceSynchronize"));
    hipSetDevice_ =
        reinterpret_cast<hipSetDevice_fn>(dlsym(hip_lib_, "hipSetDevice"));

    if (!hipHostMalloc_ || !hipHostFree_ || !hipMemcpy_ || !hipDeviceSync_) {
        NIXL_WARN << "DRAM staging: one or more HIP symbols not resolved, "
                     "staging disabled";
        dlclose(hip_lib_);
        hip_lib_ = nullptr;
        return;
    }

    available_ = true;
    NIXL_INFO << "DRAM staging: HIP runtime loaded, staging available";
}

DramStagingManager::~DramStagingManager() {
    if (hip_lib_) {
        dlclose(hip_lib_);
    }
}

void *
DramStagingManager::allocPinned(size_t bytes) {
    if (!available_) return nullptr;

    void *ptr = nullptr;
    int rc = hipHostMalloc_(&ptr, bytes, HIP_HOST_MALLOC_DEFAULT);
    if (rc != 0 || !ptr) {
        NIXL_ERROR << "DRAM staging: hipHostMalloc(" << bytes
                   << ") failed, rc=" << rc;
        return nullptr;
    }
    return ptr;
}

void
DramStagingManager::freePinned(void *ptr) {
    if (!available_ || !ptr) return;
    hipHostFree_(ptr);
}

void
DramStagingManager::copyD2H(void *host_dst, const void *gpu_src, size_t bytes) {
    if (!available_) return;
    if (hipSetDevice_) hipSetDevice_(0);
    int rc = hipMemcpy_(host_dst, gpu_src, bytes, HIP_MEMCPY_DEFAULT);
    if (rc != 0) {
        NIXL_ERROR << "DRAM staging: hipMemcpy D2H failed, rc=" << rc;
    }
}

void
DramStagingManager::copyH2D(void *gpu_dst, const void *host_src, size_t bytes) {
    if (!available_) return;
    if (hipSetDevice_) hipSetDevice_(0);
    int rc = hipMemcpy_(gpu_dst, host_src, bytes, HIP_MEMCPY_DEFAULT);
    if (rc != 0) {
        NIXL_ERROR << "DRAM staging: hipMemcpy H2D failed, rc=" << rc;
    }
}

void
DramStagingManager::sync() {
    if (!available_) return;
    hipDeviceSync_();
}
