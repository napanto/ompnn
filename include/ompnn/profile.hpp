// SPDX-License-Identifier: LGPL-3.0-only
// ompnn - per-phase profiler from host timers around synchronous regions.
//
// OpenMP target regions and omp_target_memcpy are synchronous (no `nowait` is
// used, see network.hpp), so a host clock around each region measures its
// device time plus its launch latency.  That is exactly what the study wants to
// expose for the OpenMP model; the numbers are cross-checked with nsys/rocprof.
#pragma once

#include <chrono>
#include <cstdint>

#include "ompnn/config.hpp"

namespace ompnn {

enum class Phase { H2D, D2H, Gemm, Act, Delta, BiasGrad, Update, Loss, Reg, Other };

class Profiler {
  public:
    explicit Profiler(bool enabled = false) : m_enabled(enabled) {}
    bool enabled() const { return m_enabled; }
    Profile &profile() { return m_profile; }
    const Profile &profile() const { return m_profile; }
    void reset() { m_profile.reset(); }

    /// Run `fn()` (a synchronous region) and charge its wall time to `phase`.
    template <typename F> void record(Phase phase, F &&fn, std::uint64_t bytes = 0) {
        if (!m_enabled) {
            fn();
            return;
        }
        ++m_profile.launches;
        if (phase == Phase::H2D)
            m_profile.bytes_h2d += bytes;
        else if (phase == Phase::D2H)
            m_profile.bytes_d2h += bytes;
        const auto t0 = clock::now();
        fn();
        slot(phase) += elapsed_ns(t0);
    }
    void flush() {}
    template <typename F> void timed_wait(F &&f) {
        const auto t0 = clock::now();
        f();
        m_profile.wait_ns += elapsed_ns(t0);
    }

    using clock = std::chrono::steady_clock;
    static std::uint64_t elapsed_ns(clock::time_point since) {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - since).count());
    }

  private:
    std::uint64_t &slot(Phase p) {
        switch (p) {
        case Phase::H2D: return m_profile.h2d_ns;
        case Phase::D2H: return m_profile.d2h_ns;
        case Phase::Gemm: return m_profile.gemm_ns;
        case Phase::Act: return m_profile.act_ns;
        case Phase::Delta: return m_profile.delta_ns;
        case Phase::BiasGrad: return m_profile.biasgrad_ns;
        case Phase::Update: return m_profile.update_ns;
        case Phase::Loss: return m_profile.loss_ns;
        case Phase::Reg: return m_profile.reg_ns;
        case Phase::Other:
        default: return m_profile.other_ns;
        }
    }
    bool m_enabled;
    Profile m_profile;
};

} // namespace ompnn
