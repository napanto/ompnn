// SPDX-License-Identifier: LGPL-3.0-only
// ompnn - device enumeration and selection: the OpenMP host device (CPU) plus
// every `omp target` device the runtime offers.
#pragma once

#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ompnn/config.hpp"

#include "ompnn/vendor.hpp"

namespace ompnn {

/// OpenMP's view of a compute device.  index 0 is always the host (CPU); the
/// offload devices follow with index = omp device number + 1.
struct DeviceInfo {
    int index = 0;
    std::string name;
    std::string vendor;
    std::string type; ///< "cpu" or "gpu"
    std::string backend;
    std::string platform;
    std::string driver;
    std::uint64_t global_mem_bytes = 0;
    unsigned compute_units = 0;
    bool fp64 = true;
    int omp_device = -1; ///< omp device number (-1 = host)
};

namespace detail {
inline std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) {
                std::string s = line.substr(colon + 1);
                while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
                    s.erase(s.begin());
                return s;
            }
        }
    return "CPU";
}
inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}
inline bool is_number(const std::string &s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}
} // namespace detail

inline std::string offload_backend_name() {
#if defined(OMPNN_TARGET_NVIDIA)
    return "omp-nvptx";
#elif defined(OMPNN_TARGET_AMD)
    return "omp-amdgcn";
#else
    return "omp-target";
#endif
}

inline DeviceInfo host_device() {
    DeviceInfo d;
    d.index = 0;
    d.name = detail::cpu_model() + " (OpenMP host, " + std::to_string(omp_get_max_threads()) + " threads)";
    d.vendor = "host";
    d.type = "cpu";
    d.backend = "omp";
    d.platform = "OpenMP " + std::to_string(_OPENMP);
    d.driver = std::to_string(_OPENMP);
    d.compute_units = static_cast<unsigned>(omp_get_num_procs());
    d.omp_device = -1;
    return d;
}

inline DeviceInfo offload_device(int omp_dev) {
    DeviceInfo d;
    d.index = omp_dev + 1;
    d.type = "gpu";
    d.backend = offload_backend_name();
    d.omp_device = omp_dev;
    d.name = "OpenMP target device " + std::to_string(omp_dev);
    d.platform = "OpenMP " + std::to_string(_OPENMP) + " target";
#if defined(OMPNN_TARGET_NVIDIA) || defined(OMPNN_TARGET_AMD)
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, omp_dev) == cudaSuccess) {
        d.name = p.name;
        d.global_mem_bytes = static_cast<std::uint64_t>(p.totalGlobalMem);
        d.compute_units = static_cast<unsigned>(p.multiProcessorCount);
        int drv = 0;
        (void)cudaDriverGetVersion(&drv);
        d.driver = std::to_string(drv / 1000) + "." + std::to_string((drv % 1000) / 10);
    }
#if defined(OMPNN_TARGET_NVIDIA)
    d.vendor = "NVIDIA";
#else
    d.vendor = "AMD";
#endif
#else
    d.vendor = "unknown";
#endif
    return d;
}

/// Host first, then the offload devices (none in a host-only build, even if the
/// OpenMP runtime can see accelerators: the binary carries no device code).
inline std::vector<DeviceInfo> devices() {
    std::vector<DeviceInfo> out{host_device()};
#ifdef OMPNN_HOST_ONLY
    const int n = 0;
#else
    const int n = omp_get_num_devices();
#endif
    for (int i = 0; i < n; ++i)
        out.push_back(offload_device(i));
    return out;
}

/**
 * Resolve an Options::device string to a DeviceInfo:
 *   "default"                  -> the first offload device if this build has GPU
 *                                 support and one is present, else the host
 *   "cpu" | "host"             -> the host
 *   "gpu"                      -> offload device 0
 *   "index:N" | "N"            -> devices()[N]
 *   "omp:N" | "cuda:N" | "hip:N" | "target:N" -> offload device N
 *   anything else              -> case-insensitive substring of a device name
 */
inline DeviceInfo select_device(const std::string &spec) {
    const std::string s = detail::lower(spec);
    const auto devs = devices();
    auto no_device = [&](const std::string &what) {
        std::string names;
        for (const auto &d : devs)
            names += "\n  index:" + std::to_string(d.index) + "  " + d.backend + ":" + d.type + "  " + d.name;
        return std::invalid_argument("no OpenMP device matches '" + spec + "' (" + what + "); available:" + names);
    };
    const bool have_gpu = devs.size() > 1;
    if (s.empty() || s == "default" || s == "auto")
        return have_gpu ? devs[1] : devs[0];
    if (s == "cpu" || s == "host")
        return devs[0];
    if (s == "gpu" || s == "accelerator") {
        if (!have_gpu)
            throw no_device("no offload device (build without GPU support, or none present)");
        return devs[1];
    }
    std::string key = s, arg;
    const auto colon = s.find(':');
    if (colon != std::string::npos) {
        key = s.substr(0, colon);
        arg = s.substr(colon + 1);
    }
    if (detail::is_number(s) || (key == "index" && detail::is_number(arg))) {
        const std::size_t n = std::stoul(detail::is_number(s) ? s : arg);
        if (n >= devs.size())
            throw no_device("index out of range");
        return devs[n];
    }
    if ((key == "omp" || key == "cuda" || key == "hip" || key == "target") && detail::is_number(arg)) {
        const std::size_t n = std::stoul(arg) + 1;
        if (n >= devs.size())
            throw no_device("offload device number out of range");
        return devs[n];
    }
    if ((key == "omp" || key == "cuda" || key == "hip" || key == "target") && arg == "gpu" && have_gpu)
        return devs[1];
    for (const auto &d : devs)
        if (detail::lower(d.name).find(s) != std::string::npos)
            return d;
    throw no_device("no name match");
}

} // namespace ompnn
