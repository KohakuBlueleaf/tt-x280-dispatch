// ttx_bench - X280 protocol and dispatch-model bench.
// Usage: ttx_bench compute [pll_mhz ...]   (run with TT_VISIBLE_DEVICES=<one
// card>, TT_METAL_RUNTIME_ROOT set)
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "x280/device/x280_runtime.hpp"
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/noc_id.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/io_window_config.hpp>

using namespace x280;
using tt::tt_metal::distributed::MeshDevice;

namespace {

constexpr double kMtimeHz = 50e6;
const std::string kImageDir = X280_IMAGE_DIR;
uint64_t g_run = 0x7474780000000000ull;  // "ttx"
// Per-process nonce for L1 check-in/gather tags: L1 contents survive between processes,
// so tags must not repeat.
const uint32_t g_tag_nonce = std::random_device{}();

// A check-in tag unique to this process and run (never 0).
uint32_t fresh_tag(uint64_t run) {
    const uint32_t t =
        (g_tag_nonce ^ static_cast<uint32_t>(run * 0x9E3779B1u)) & 0xFFFFu;
    return t ? t : 1u;
}

struct Mode {
    uint64_t id;
    const char* name;
    double ops_per_iter;  // per hart, counting an FMA as 2
    uint64_t iters;
};

// VLEN=512: e32 LMUL=1 -> 16 lanes; LMUL=8 -> 128 lanes.
const Mode kModes[] = {
    {0, "fp32 vfmacc m1 x8", 8 * 16 * 2.0, 2'000'000},
    {1, "fp32 vfmacc m8 x2", 2 * 128 * 2.0, 1'000'000},
    {2, "int32 vmacc m1 x8", 8 * 16 * 2.0, 2'000'000},
    {3, "scalar fp32 fmadd x8", 8 * 2.0, 2'000'000},
};

// Boots every X280 tile into the idle image if it is not idle.
void boot_all(X280Runtime& rt) {
    bool all_idle = true;
    for (const auto& id : rt.tile_ids()) {
        all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
    }
    if (all_idle) {
        return;
    }
    MultiTileBootRequest req;
    req.tiles = rt.tile_ids();
    req.idle_image_path = kImageDir + "/firmware-lim-idle.bin";
    req.allow_reset_release = true;
    const auto booted = rt.boot_tiles(req);
    if (!booted.ready()) {
        throw std::runtime_error("boot_tiles failed: " + booted.diagnostic);
    }
}

// Runs one mode on the first `ntiles` tiles concurrently; returns per-hart ticks
// (tile-major).
std::vector<uint64_t> run_mode(X280Runtime& rt, const Mode& m, size_t ntiles) {
    const auto ids = rt.tile_ids();
    std::vector<SessionId> sessions;
    std::vector<uint64_t> runs;
    for (size_t t = 0; t < ntiles; t++) {
        X280Tile& tile = rt.tile(ids[t]);
        const uint64_t run = ++g_run;
        const uint64_t args_addr = tile.available_lim().begin;
        std::vector<uint8_t> block(64, 0);
        const uint64_t fields[3] = {run, m.id, m.iters};
        std::memcpy(block.data(), fields, sizeof(fields));
        tile.write_available_lim(args_addr, block);
        ActiveRunRequest req;
        req.image_path = kImageDir + "/ttx-compute.bin";
        req.run_value = run;
        req.args_addr = args_addr;
        req.args_bytes = block.size();
        const auto enq = rt.enqueue(tile, req);
        if (!enq.session) {
            throw std::runtime_error("enqueue failed: " + describe(enq));
        }
        sessions.push_back(enq.session);
        runs.push_back(run);
    }
    std::vector<uint64_t> ticks;
    for (size_t t = 0; t < sessions.size(); t++) {
        const auto res =
            rt.session(sessions[t]).wait_until_result_ready(std::chrono::seconds(120));
        if (!res.succeeded()) {
            throw std::runtime_error("wait failed: " + describe(res));
        }
        for (const auto& h : res.hartstates) {
            ticks.push_back(h.state.result);
        }
        const auto cleaned = rt.cleanup(sessions[t], std::chrono::seconds(5));
        if (!cleaned.cleaned_up()) {
            throw std::runtime_error("cleanup failed: " + describe(cleaned));
        }
    }
    return ticks;
}

// X280 vector and scalar compute throughput at each PLL setting.
void compute(X280Runtime& rt, const std::vector<uint32_t>& plls) {
    const auto ids = rt.tile_ids();
    std::printf("# %zu tiles x 4 harts; mtime %.0f MHz\n", ids.size(), kMtimeHz / 1e6);
    for (uint32_t mhz : plls) {
        if (mhz) {
            rt.set_pll(ids.front().chip_id, mhz);
        }
        std::printf("## PLL %u MHz\n", mhz ? mhz : kSafePllMhz);
        for (const Mode& m : kModes) {
            for (size_t ntiles : {size_t{1}, ids.size()}) {
                const auto ticks = run_mode(rt, m, ntiles);
                uint64_t worst = 0;
                for (uint64_t t : ticks) {
                    worst = std::max(worst, t);
                }
                const double sec = worst / kMtimeHz;
                const double ops = m.ops_per_iter * m.iters * ticks.size();
                const double per_hart = m.ops_per_iter * m.iters / sec;
                const double clk = (mhz ? mhz : kSafePllMhz) * 1e6;
                std::printf(
                    "%-22s tiles=%zu harts=%2zu  %8.2f Gop/s total  %7.2f Gop/s/hart  "
                    "%5.2f op/cycle/hart\n",
                    m.name,
                    ntiles,
                    ticks.size(),
                    ops / sec / 1e9,
                    per_hart / 1e9,
                    per_hart / clk
                );
                std::fflush(stdout);
            }
        }
    }
}

// Tensix core (0,0) <-> X280 tile 0 hart 0 ping-pong; prints mean round-trip latency
// for several n.
void pingpong(X280Runtime& rt, const std::shared_ptr<MeshDevice>& mesh) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const auto ids = rt.tile_ids();
    X280Tile& tile = rt.tile(ids.front());
    const tm::CoreCoord core{0, 0};
    const auto tensix = X280Runtime::virtual_core_from_logical_core(
        *mesh, core, tt::CoreType::WORKER, tm::NOC::NOC_0
    );
    const auto l2cpu = rt.virtual_core_from_tile(ids.front(), tm::NOC::NOC_0);
    const uint64_t args_addr = tile.available_lim().begin;
    const uint64_t mailbox = args_addr + 0x100;
    std::printf(
        "# tensix (0,0) -> translated (%u,%u); L2CPU tile -> (%u,%u); mailbox 0x%llx\n",
        tensix.x(),
        tensix.y(),
        l2cpu.x(),
        l2cpu.y(),
        (unsigned long long)mailbox
    );
    const std::string kernel = std::string(TTX_KERNEL_DIR) + "/protocol/pingpong.cpp";

    for (uint64_t n : {uint64_t{1000}, uint64_t{10000}, uint64_t{100000}}) {
        tile.write_available_lim(mailbox, std::vector<uint8_t>(64, 0));
        const uint64_t run = ++g_run;
        std::vector<uint8_t> block(64, 0);
        const uint64_t fields[5] = {run, mailbox, n, tensix.x(), tensix.y()};
        std::memcpy(block.data(), fields, sizeof(fields));
        tile.write_available_lim(args_addr, block);
        ActiveRunRequest req;
        req.image_path = kImageDir + "/ttx-pingpong.bin";
        req.run_value = run;
        req.args_addr = args_addr;
        req.args_bytes = block.size();
        const auto enq = rt.enqueue(tile, req);
        if (!enq.session) {
            throw std::runtime_error("enqueue failed: " + describe(enq));
        }

        tm::Program program = tm::CreateProgram();
        tm::CreateCircularBuffer(
            program,
            core,
            tm::CircularBufferConfig(
                256, {{0, tt::DataFormat::UInt32}}
            ).set_page_size(0, 256)
        );
        const auto k = tm::CreateKernel(
            program,
            kernel,
            core,
            tm::DataMovementConfig{
                .processor = tm::DataMovementProcessor::RISCV_0,
                .noc = tm::NOC::RISCV_0_default
            }
        );
        tm::SetRuntimeArgs(
            program,
            k,
            core,
            {l2cpu.x(),
             l2cpu.y(),
             static_cast<uint32_t>(mailbox),
             static_cast<uint32_t>(n)}
        );
        dist::MeshWorkload workload;
        workload.add_program(
            dist::MeshCoordinateRange(mesh->shape()), std::move(program)
        );
        const auto t0 = std::chrono::steady_clock::now();
        dist::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);

        const auto res =
            rt.session(enq.session).wait_until_result_ready(std::chrono::seconds(60));
        dist::Finish(mesh->mesh_command_queue());
        const double host_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        const auto cleaned = rt.cleanup(enq.session, std::chrono::seconds(5));
        if (!res.succeeded() || !cleaned.cleaned_up()) {
            throw std::runtime_error(
                "pingpong failed: " + describe(res) + " / " + describe(cleaned)
            );
        }
        const uint64_t ticks = res.hartstates[0].state.result;
        if (ticks >= ~16ull) {
            throw std::runtime_error(
                "pingpong image error code " + std::to_string(~ticks)
            );
        }
        const double us = ticks / kMtimeHz * 1e6 / static_cast<double>(n - 1);
        std::printf(
            "pingpong n=%-7llu  %.3f us/round-trip (X280 mtime)  host wall %.1f ms "
            "incl. launch\n",
            (unsigned long long)n,
            us,
            host_s * 1e3
        );
        std::fflush(stdout);
    }
}

struct SchedCase {
    const char* name;
    uint32_t lx0, ly0, lx1,
        ly1;  // logical core rectangle(s) (second optional, lx1 < lx0 = none)
    uint32_t mx0, my0, mx1, my1;
};

// X280 scheduler broadcasting commands to persistent Tensix kernels; prints mean
// broadcast+gather round time.
void sched(X280Runtime& rt, const std::shared_ptr<MeshDevice>& mesh, uint64_t n) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const auto ids = rt.tile_ids();
    X280Tile& tile = rt.tile(ids.front());
    const auto l2cpu = rt.virtual_core_from_tile(ids.front(), tm::NOC::NOC_0);
    const uint64_t args_addr = tile.available_lim().begin;
    const uint64_t mailbox = args_addr + 0x100;
    const uint64_t acks = args_addr + 0x400;  // 16 B per ack slot, up to 128 workers
    const std::string kernel = std::string(TTX_KERNEL_DIR) + "/protocol/sched.cpp";
    auto tr = [&](uint32_t x, uint32_t y) {
        return X280Runtime::virtual_core_from_logical_core(
            *mesh, tm::CoreCoord{x, y}, tt::CoreType::WORKER, tm::NOC::NOC_0
        );
    };
    // Logical rectangles: [0] always, [1] optional (x1 < x0 means none).
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t last_x = static_cast<uint32_t>(grid.x) - 1;
    const SchedCase cases[] = {
        {"1 core (unicast)", 0, 0, 0, 0, 1, 0, 0, 0},
        {"7 cores (row mcast)", 0, 0, 6, 0, 1, 0, 0, 0},
        {"70 cores (block mcast)", 0, 0, 6, 9, 1, 0, 0, 0},
        {"full grid (2 mcast windows)", 0, 0, 6, 9, 7, 0, last_x, 9},
    };
    std::vector<size_t> which = {0, 1, 2, 3};
    if (const char* sel = std::getenv("TTX_CASES")) {
        which.clear();
        for (const char* p = sel; *p; p++) {
            if (*p >= '0' && *p <= '3') {
                which.push_back(static_cast<size_t>(*p - '0'));
            }
        }
    }
    std::printf(
        "# compute grid %zux%zu\n",
        static_cast<size_t>(grid.x),
        static_cast<size_t>(grid.y)
    );
    for (size_t ci : which) {
        const SchedCase& c = cases[ci];
        std::vector<std::array<uint32_t, 4>> lrects = {{c.lx0, c.ly0, c.lx1, c.ly1}};
        if (c.mx1 >= c.mx0) {
            lrects.push_back({c.mx0, c.my0, c.mx1, c.my1});
        }
        std::vector<tm::CoreCoord> cores;
        std::vector<tm::CoreRange> ranges;
        uint32_t trect[2][4] = {};
        for (size_t r = 0; r < lrects.size(); r++) {
            const auto& q = lrects[r];
            ranges.emplace_back(tm::CoreCoord{q[0], q[1]}, tm::CoreCoord{q[2], q[3]});
            for (uint32_t y = q[1]; y <= q[3]; y++) {
                for (uint32_t x = q[0]; x <= q[2]; x++) {
                    cores.push_back({x, y});
                }
            }
            const auto a = tr(q[0], q[1]);
            const auto b = tr(q[2], q[3]);
            trect[r][0] = a.x();
            trect[r][1] = a.y();
            trect[r][2] = b.x();
            trect[r][3] = b.y();
        }
        const tm::CoreRangeSet crs(ranges);

        tile.write_available_lim(mailbox, std::vector<uint8_t>(64, 0));
        tile.write_available_lim(acks, std::vector<uint8_t>(2048, 0));
        const uint64_t run = ++g_run;
        std::vector<uint8_t> block(128, 0);
        const uint64_t head[6] = {run, mailbox, acks, n, cores.size(), lrects.size()};
        std::memcpy(block.data(), head, sizeof(head));
        std::memcpy(block.data() + sizeof(head), trect, sizeof(trect));
        tile.write_available_lim(args_addr, block);
        ActiveRunRequest req;
        req.image_path = kImageDir + "/ttx-sched.bin";
        req.run_value = run;
        req.args_addr = args_addr;
        req.args_bytes = block.size();
        const auto enq = rt.enqueue(tile, req);
        if (!enq.session) {
            throw std::runtime_error("enqueue failed: " + describe(enq));
        }

        tm::Program program = tm::CreateProgram();
        tm::CreateCircularBuffer(
            program,
            crs,
            tm::CircularBufferConfig(
                256, {{0, tt::DataFormat::UInt32}}
            ).set_page_size(0, 256)
        );
        const auto k = tm::CreateKernel(
            program,
            kernel,
            crs,
            tm::DataMovementConfig{
                .processor = tm::DataMovementProcessor::RISCV_0,
                .noc = tm::NOC::RISCV_0_default
            }
        );
        for (uint32_t i = 0; i < cores.size(); i++) {
            tm::SetRuntimeArgs(
                program,
                k,
                cores[i],
                {l2cpu.x(),
                 l2cpu.y(),
                 static_cast<uint32_t>(mailbox),
                 static_cast<uint32_t>(acks),
                 i,
                 static_cast<uint32_t>(n),
                 i == 0 ? 1u : 0u}
            );
        }
        dist::MeshWorkload workload;
        workload.add_program(
            dist::MeshCoordinateRange(mesh->shape()), std::move(program)
        );
        dist::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);
        const auto res =
            rt.session(enq.session).wait_until_result_ready(std::chrono::seconds(20));
        const auto cleaned = rt.cleanup(enq.session, std::chrono::seconds(5));
        dist::Finish(mesh->mesh_command_queue());
        if (!res.succeeded() || !cleaned.cleaned_up()) {
            std::printf(
                "%-30s FAILED: %s / %s\n",
                c.name,
                describe(res).c_str(),
                describe(cleaned).c_str()
            );
            std::fflush(stdout);
            continue;
        }
        const uint64_t ticks = res.hartstates[0].state.result;
        if (ticks >= ~16ull) {
            throw std::runtime_error(
                "sched image error code " + std::to_string(~ticks)
            );
        }
        const double us = ticks / kMtimeHz * 1e6 / static_cast<double>(n - 1);
        std::printf(
            "%-30s K=%3zu  %.3f us per broadcast+all-ack round  (%.2f M rounds/s)\n",
            c.name,
            cores.size(),
            us,
            1.0 / us
        );
        std::fflush(stdout);
    }
}

struct DispCase {
    const char* name;
    uint32_t sig;     // 0 flat acks, 1 row-leader aggregation, 2 Tensix-only barrier
    uint32_t scan;    // X280: 0 per-slot spin, 1 sweep, 2 watch slot 0
    uint32_t stride;  // ack slot stride, bytes
    uint32_t depth;   // command ring depth (1 = synchronous)
    uint32_t host;    // 1: commands fed by the host through a LIM ring
    uint32_t grp =
        0;  // aggregation groups: 0 rows, 1 columns, 2 rows within each panel
    uint32_t lpos =
        0;  // group leader at the low (0: west / north) or high (1: east / south) end
    uint32_t noc = 0;    // NOC used by every worker (0: east/south, 1: west/north)
    uint32_t relay = 0;  // 1: X280 unicasts to core (0,0), which forwards with a
                         // non-posted Tensix multicast
};

// Dispatch-model experiments (E6) on the full compute grid; TTX_DCASES selects cases,
// TTX_WORK = wall-clock ticks per command.
void disp(X280Runtime& rt, const std::shared_ptr<MeshDevice>& mesh, uint64_t n) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const DispCase cases[] = {
        {"flat, per-slot spin (E4)", 0, 0, 16, 1, 0},
        {"flat, sweep", 0, 1, 16, 1, 0},
        {"flat, sweep, packed acks", 0, 1, 4, 1, 0},
        {"row leaders, sweep", 1, 1, 16, 1, 0},
        {"row leaders, sweep, packed", 1, 1, 4, 1, 0},
        {"Tensix-only barrier", 2, 2, 16, 1, 0},
        {"flat packed, ring depth 4", 0, 1, 4, 4, 0},
        {"row leaders packed, ring 4", 1, 1, 4, 4, 0},
        {"row leaders packed, ring 16", 1, 1, 4, 16, 0},
        {"HOST-fed, row leaders, sync", 1, 1, 4, 1, 1},
        {"HOST-fed, row leaders, ring 4", 1, 1, 4, 4, 1},
        {"rows, leader W, NOC1", 1, 1, 4, 1, 0, 0, 0, 1},
        {"rows, leader E, NOC0", 1, 1, 4, 1, 0, 0, 1, 0},
        {"rows, leader E, NOC1", 1, 1, 4, 1, 0, 0, 1, 1},
        {"cols, leader N, NOC0", 1, 1, 4, 1, 0, 1, 0, 0},
        {"cols, leader N, NOC1", 1, 1, 4, 1, 0, 1, 0, 1},
        {"cols, leader S, NOC0", 1, 1, 4, 1, 0, 1, 1, 0},
        {"cols, leader S, NOC1", 1, 1, 4, 1, 0, 1, 1, 1},
        {"panel rows, leader W, NOC1", 1, 1, 4, 1, 0, 2, 0, 1},
        {"panel rows, leader E, NOC0", 1, 1, 4, 1, 0, 2, 1, 0},
        {"panel rows, leader W, NOC0", 1, 1, 4, 1, 0, 2, 0, 0},
        {"rows, leader E, NOC0, ring 4", 1, 1, 4, 4, 0, 0, 1, 0},
        {"cols, leader S, NOC0, ring 4", 1, 1, 4, 4, 0, 1, 1, 0},
        {"panel rows, leader E, NOC0, ring 4", 1, 1, 4, 4, 0, 2, 1, 0},
        {"rows, leader W, NOC1, ring 4", 1, 1, 4, 4, 0, 0, 0, 1},
        {"rows, 16B acks, ring 4", 1, 1, 16, 4, 0},
        {"rows, 16B acks, ring 16", 1, 1, 16, 16, 0},
        {"cols, leader N, NOC0, 16B acks", 1, 1, 16, 1, 0, 1, 0, 0},
        {"cols, leader N, NOC0, 16B, ring 4", 1, 1, 16, 4, 0, 1, 0, 0},
        {"rows, WRITE gather, sync", 3, 1, 16, 1, 0},
        {"rows, WRITE gather, ring 4", 3, 1, 16, 4, 0},
        {"cols, WRITE gather, sync", 3, 1, 16, 1, 0, 1, 0, 0},
        {"cols, WRITE gather, ring 4", 3, 1, 16, 4, 0, 1, 0, 0},
        {"rows, WRITE gather, ring 16", 3, 1, 16, 16, 0},
        {"RELAY, flat acks, sync", 0, 1, 16, 1, 0, 0, 0, 0, 1},
        {"RELAY, cols WRITE gather, sync", 3, 1, 16, 1, 0, 1, 0, 0, 1},
        {"RELAY, cols WRITE gather, ring 4", 3, 1, 16, 4, 0, 1, 0, 0, 1},
        {"RELAY, rows WRITE gather, ring 16", 3, 1, 16, 16, 0, 0, 0, 0, 1},
        {"RELAY, cols atomic, ring 4", 1, 1, 16, 4, 0, 1, 0, 0, 1},
    };
    constexpr size_t kNumCases = sizeof(cases) / sizeof(cases[0]);
    std::vector<size_t> which;
    if (const char* sel = std::getenv("TTX_DCASES")) {
        std::string s(sel);
        size_t p = 0;
        while (p < s.size()) {
            const size_t q = s.find(',', p);
            which.push_back(std::stoul(s.substr(p, q - p)));
            p = q == std::string::npos ? s.size() : q + 1;
        }
    } else {
        for (size_t i = 0; i < kNumCases; i++) {
            which.push_back(i);
        }
    }
    const uint32_t work =
        std::getenv("TTX_WORK") ? std::atoi(std::getenv("TTX_WORK")) : 0;
    // X280 command-window NOC ordering: 0 default (non-posted), 1 strict, 2 posted.
    const uint64_t order =
        std::getenv("TTX_ORDER") ? std::atoi(std::getenv("TTX_ORDER")) : 2;
    // Command ring slot stride in bytes: 16 (four slots per 64 B line) or 64 (one slot
    // per line).
    const uint64_t ring_stride =
        std::getenv("TTX_RSTRIDE") ? std::atoi(std::getenv("TTX_RSTRIDE")) : 16;
    // X280 command store: 0 = 32-bit scalar store, 1 = one 64 B vector store (needs
    // TTX_RSTRIDE=64).
    const uint64_t store_mode =
        std::getenv("TTX_STORE") ? std::atoi(std::getenv("TTX_STORE")) : 0;
    // X280 idle gap after each command issue, mtime ticks (20 ns); TTX_SWAPWIN=1 issues
    // the right panel window first.
    const uint64_t issue_gap =
        std::getenv("TTX_GAP") ? std::atoi(std::getenv("TTX_GAP")) : 0;
    const bool swap_win =
        std::getenv("TTX_SWAPWIN") && std::atoi(std::getenv("TTX_SWAPWIN"));

    const auto ids = rt.tile_ids();
    X280Tile& tile = rt.tile(ids.front());
    const auto l2cpu = rt.virtual_core_from_tile(ids.front(), tm::NOC::NOC_0);
    const uint64_t args_addr = tile.available_lim().begin;
    const uint64_t mailbox = args_addr + 0x100, acks = args_addr + 0x400,
                   hring = args_addr + 0xC00, hdone = args_addr + 0x1000;
    const std::string kernel = std::string(TTX_KERNEL_DIR) + "/protocol/disp.cpp";
    auto tr = [&](uint32_t x, uint32_t y) {
        return X280Runtime::virtual_core_from_logical_core(
            *mesh, tm::CoreCoord{x, y}, tt::CoreType::WORKER, tm::NOC::NOC_0
        );
    };
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t gx = grid.x, gy = grid.y, K = gx * gy;
    // Panels: logical x 0..6 and 7..gx-1 (translated columns 1-7 and 10+), never
    // multicast across 8/9. Multicast panels by logical column: [0, split) and [split,
    // gx). TTX_SPLIT=6 gives physically contiguous Tensix-only rectangles on our
    // harvested p150a (translated 1-6 -> NOC0 1-6, translated 7-14 -> NOC0 11-16); the
    // old split 7 makes the first rectangle span NOC0 1-11 (harvested 7/10, L2CPU 8,
    // DRAM 9).
    const uint32_t split =
        std::getenv("TTX_SPLIT") ? std::atoi(std::getenv("TTX_SPLIT")) : 7;
    const uint32_t px[2][2] = {{0, split - 1}, {split, gx - 1}};
    uint32_t trect[2][4];
    for (int r = 0; r < 2; r++) {
        const auto a = tr(px[r][0], 0), b = tr(px[r][1], gy - 1);
        trect[r][0] = a.x();
        trect[r][1] = a.y();
        trect[r][2] = b.x();
        trect[r][3] = b.y();
    }
    std::printf(
        "# grid %ux%u = %u cores, n=%llu, work=%u ticks/cmd, window ordering %llu\n",
        gx,
        gy,
        K,
        (unsigned long long)n,
        work,
        (unsigned long long)order
    );

    for (size_t ci : which) {
        if (ci >= kNumCases) {
            continue;
        }
        const DispCase& c = cases[ci];
        // Group of core (x, y) for sig 1: {group id, member count, leader logical x,
        // leader logical y}.
        auto group = [&](uint32_t x, uint32_t y) -> std::array<uint32_t, 4> {
            if (c.grp == 1) {
                return {x, gy, x, c.lpos ? gy - 1 : 0};
            }
            if (c.grp == 2) {
                const uint32_t p = x <= px[0][1] ? 0 : 1;
                return {
                    y * 2 + p, px[p][1] - px[p][0] + 1, c.lpos ? px[p][1] : px[p][0], y
                };
            }
            return {y, gx, c.lpos ? gx - 1 : 0, y};
        };
        const uint32_t groups = c.grp == 1 ? gx : c.grp == 2 ? 2 * gy : gy;
        const uint32_t slots = c.sig == 0 ? K : (c.sig == 1 || c.sig == 3) ? groups : 1;
        tile.write_available_lim(mailbox, std::vector<uint8_t>(64, 0));
        tile.write_available_lim(acks, std::vector<uint8_t>(2048, 0));
        tile.write_available_lim(hring, std::vector<uint8_t>(0x440, 0));
        tile.write_available_lim(acks + 0x1000, std::vector<uint8_t>(0x800, 0));
        const uint64_t run = ++g_run;
        const uint32_t tag = fresh_tag(run);
        std::vector<uint8_t> block(192, 0);
        const uint64_t head[6] = {run, mailbox, acks, n, slots, c.relay ? 1u : 2u};
        std::memcpy(block.data(), head, sizeof(head));
        if (c.relay) {  // one unicast window aimed at the relay core (0,0)
            const auto rc = tr(0, 0);
            const uint32_t urect[2][4] = {
                {static_cast<uint32_t>(rc.x()),
                 static_cast<uint32_t>(rc.y()),
                 static_cast<uint32_t>(rc.x()),
                 static_cast<uint32_t>(rc.y())},
                {0, 0, 0, 0}
            };
            std::memcpy(block.data() + sizeof(head), urect, sizeof(urect));
        } else if (swap_win) {
            const uint32_t sw[2][4] = {
                {trect[1][0], trect[1][1], trect[1][2], trect[1][3]},
                {trect[0][0], trect[0][1], trect[0][2], trect[0][3]}
            };
            std::memcpy(block.data() + sizeof(head), sw, sizeof(sw));
        } else {
            std::memcpy(block.data() + sizeof(head), trect, sizeof(trect));
        }
        const uint64_t tail[10] = {
            c.scan,
            c.depth,
            c.stride,
            c.host ? hring : 0,
            16,
            c.host ? hdone : 0,
            order,
            ring_stride,
            store_mode,
            issue_gap
        };
        std::memcpy(block.data() + sizeof(head) + sizeof(trect), tail, sizeof(tail));
        tile.write_available_lim(args_addr, block);
        ActiveRunRequest req;
        req.image_path = kImageDir + "/ttx-sequencer.bin";
        req.run_value = run;
        req.args_addr = args_addr;
        req.args_bytes = block.size();
        const auto enq = rt.enqueue(tile, req);
        if (!enq.session) {
            throw std::runtime_error("enqueue failed: " + describe(enq));
        }

        tm::Program program = tm::CreateProgram();
        const tm::CoreRange all({0, 0}, {gx - 1, gy - 1});
        tm::CreateCircularBuffer(
            program,
            all,
            tm::CircularBufferConfig(
                2048, {{0, tt::DataFormat::UInt32}}
            ).set_page_size(0, 2048)
        );
        const uint32_t sem = tm::CreateSemaphore(program, all, 0);
        const auto k = tm::CreateKernel(
            program,
            kernel,
            all,
            tm::DataMovementConfig{
                .processor = tm::DataMovementProcessor::RISCV_0,
                .noc = c.noc ? tm::NOC::RISCV_1_default : tm::NOC::RISCV_0_default
            }
        );
        const auto lead = tr(0, 0);
        for (uint32_t y = 0; y < gy; y++) {
            for (uint32_t x = 0; x < gx; x++) {
                const uint32_t idx = y * gx + x;
                const auto g = group(x, y);
                const auto gl = (c.sig == 1 || c.sig == 3) ? tr(g[2], g[3]) : lead;
                const bool hier = c.sig == 1 || c.sig == 3;
                const bool is_gl = hier && x == g[2] && y == g[3];
                const uint32_t pan = x <= px[0][1] ? 0 : 1;
                const uint32_t pos = c.grp == 1 ? y : c.grp == 2 ? x - px[pan][0] : x;
                const uint32_t lpos_i = c.grp == 1   ? g[3]
                                        : c.grp == 2 ? g[2] - px[pan][0]
                                                     : g[2];
                const uint32_t rank = pos < lpos_i ? pos : pos - 1;
                std::vector<uint32_t> args = {
                    static_cast<uint32_t>(l2cpu.x()),
                    static_cast<uint32_t>(l2cpu.y()),
                    static_cast<uint32_t>(mailbox),
                    static_cast<uint32_t>(acks),
                    idx,
                    static_cast<uint32_t>(n),
                    idx == 0 ? 1u : 0u,
                    c.sig,
                    c.stride,
                    c.depth,
                    work,
                    static_cast<uint32_t>(gl.x()),
                    static_cast<uint32_t>(gl.y()),
                    hier ? g[1] : K,
                    g[0],
                    is_gl ? 1u : 0u,
                    sem
                };
                for (int r = 0; r < 2; r++) {
                    const uint32_t nd = (px[r][1] - px[r][0] + 1) * gy;
                    args.insert(
                        args.end(),
                        {trect[r][0], trect[r][1], trect[r][2], trect[r][3], nd}
                    );
                }
                args.push_back(0);
                args.push_back(hier && !is_gl ? rank : 0);
                args.push_back(tag);
                args.push_back(static_cast<uint32_t>(ring_stride));
                args.push_back((c.relay && idx == 0) ? 1u : 0u);
                tm::SetRuntimeArgs(program, k, tm::CoreCoord{x, y}, args);
            }
        }
        dist::MeshWorkload workload;
        workload.add_program(
            dist::MeshCoordinateRange(mesh->shape()), std::move(program)
        );
        dist::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, false);

        double host_us = 0, host_write_us = 0;
        if (c.host) {
            const uint32_t hdepth = 16;
            // TTX_HOSTRAW=1: the runtime transport's own cluster write/read, without
            // its per-call state inspection.
            const bool raw =
                std::getenv("TTX_HOSTRAW") && std::atoi(std::getenv("TTX_HOSTRAW"));
            const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
            const tt_cxy_pair l2core(ids.front().chip_id, l2cpu.x(), l2cpu.y());
            auto rd_done = [&]() {
                uint32_t d;
                if (raw) {
                    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
                    cluster.read_core(&d, 4, l2core, hdone);
                } else {
                    const auto v = tile.read_available_lim(hdone, 64);
                    std::memcpy(&d, v.data(), 4);
                }
                return d;
            };
            auto wr_slot = [&](uint64_t addr, const std::vector<uint8_t>& s) {
                if (raw) {
                    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
                    cluster.write_core(s.data(), 4, l2core, addr);
                } else {
                    tile.write_available_lim(addr, s);
                }
            };
            std::vector<uint8_t> slot(64, 0);
            uint32_t done = 0;
            const auto t0 = std::chrono::steady_clock::now();
            double wsum = 0;
            for (uint32_t i = 1; i <= n; i++) {
                while (i - done > hdepth - 1) {
                    done = rd_done();
                }
                std::memcpy(slot.data(), &i, 4);
                const auto w0 = std::chrono::steady_clock::now();
                wr_slot(hring + 64 * (i % hdepth), slot);
                wsum +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - w0)
                        .count();
            }
            while (done < n) {
                done = rd_done();
            }
            host_us =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count() *
                1e6 / n;
            host_write_us = wsum * 1e6 / n;
        }
        const auto res =
            rt.session(enq.session).wait_until_result_ready(std::chrono::seconds(30));
        const auto cleaned = rt.cleanup(enq.session, std::chrono::seconds(5));
        dist::Finish(mesh->mesh_command_queue());
        if (!res.succeeded() || !cleaned.cleaned_up()) {
            auto rd32 = [&](uint64_t base, size_t bytes) {
                const auto v = tile.read_available_lim(base, bytes);
                std::vector<uint32_t> w(bytes / 4);
                std::memcpy(w.data(), v.data(), bytes);
                return w;
            };
            const auto mbw = rd32(mailbox, 64);
            const auto ackw = rd32(acks, 2048);
            std::printf(
                "%-34s FAILED (%s); X280 last issued cmd %u\n   group acks:",
                c.name,
                res.succeeded() ? "cleanup" : "timeout",
                mbw[4]
            );
            for (uint32_t j = 0; j < slots; j++) {
                std::printf(" %u", ackw[j * (c.stride / 4)]);
            }
            std::printf("\n");
            // Per-core post-mortem: stage 1 = waiting for a command, 2 = leader waiting
            // on a member write, 3 = leader waiting on member atomics.
            const auto dg = rd32(acks + 0x1000, 0x800);
            uint32_t counts[4] = {};
            for (uint32_t i = 0; i < K; i++) {
                const uint32_t tagw = dg[i * 4];
                if ((tagw >> 16) != 0xD1A0) {
                    continue;
                }
                const uint32_t st = tagw & 0xF;
                counts[st < 4 ? st : 0]++;
                if (counts[st < 4 ? st : 0] <= 200) {
                    const auto t = tr(i % gx, i / gx);
                    std::printf(
                        "   core (%u,%u) [noc %u,%u] stage %u: expected %u, saw %u "
                        "(aux %u)\n",
                        i % gx,
                        i / gx,
                        static_cast<uint32_t>(t.x()),
                        static_cast<uint32_t>(t.y()),
                        st,
                        dg[i * 4 + 1],
                        dg[i * 4 + 2],
                        dg[i * 4 + 3]
                    );
                }
            }
            std::printf(
                "   diag totals: waiting-for-command %u, leader-waiting-write %u, "
                "leader-waiting-atomic %u\n",
                counts[1],
                counts[2],
                counts[3]
            );
            std::fflush(stdout);
            continue;
        }
        const uint64_t ticks = res.hartstates[0].state.result;
        if (ticks >= ~16ull) {
            std::printf("%-34s image error %llu\n", c.name, (unsigned long long)~ticks);
            continue;
        }
        const double us =
            ticks / kMtimeHz * 1e6 / static_cast<double>(c.sig == 2 ? n - 1 : n);
        std::printf("%-34s slots=%3u  %.3f us/cmd", c.name, slots, us);
        if (c.host) {
            std::printf(
                "  | host wall %.2f us/cmd (LIM write %.2f us)", host_us, host_write_us
            );
        }
        std::printf("\n");
        std::fflush(stdout);
    }
}

// E6b: Tensix atomic-increment stress without the X280. Members send k increments each
// to one leader semaphore; the leader reports the final count through DRAM. TTX_K
// (default 100000), TTX_REPS (default 3).
void atomic(const std::shared_ptr<MeshDevice>& mesh) {
    namespace tm = tt::tt_metal;
    namespace dist = tt::tt_metal::distributed;
    const uint32_t k = std::getenv("TTX_K") ? std::atoi(std::getenv("TTX_K")) : 100000;
    const int reps = std::getenv("TTX_REPS") ? std::atoi(std::getenv("TTX_REPS")) : 3;
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t gx = grid.x, gy = grid.y;
    auto tr = [&](uint32_t x, uint32_t y) {
        return X280Runtime::virtual_core_from_logical_core(
            *mesh, tm::CoreCoord{x, y}, tt::CoreType::WORKER, tm::NOC::NOC_0
        );
    };
    dist::DeviceLocalBufferConfig dram{
        .page_size = 32, .buffer_type = tm::BufferType::DRAM
    };
    auto out = dist::MeshBuffer::create(
        dist::ReplicatedBufferConfig{.size = 32}, dram, mesh.get()
    );
    const std::string kernel = std::string(TTX_KERNEL_DIR) + "/protocol/atomic.cpp";

    struct Case {
        const char* name;
        uint32_t x0, y0, x1, y1, lx, ly, noc;
        uint32_t noise =
            0;  // 1: core (6, gy-1) multicasts k words into the left panel meanwhile
    };

    const Case cases[] = {
        {"left panel + MCAST noise -> (0,0)", 0, 0, 6, gy - 1, 0, 0, 0, 1},
        {"full grid + MCAST noise -> (0,0)", 0, 0, gx - 1, gy - 1, 0, 0, 0, 1},
        {"full grid -> (0,0), NOC0", 0, 0, gx - 1, gy - 1, 0, 0, 0},
        {"full grid -> (0,0), NOC1", 0, 0, gx - 1, gy - 1, 0, 0, 1},
        {"row 1 (12 cores) -> (0,1), NOC0", 0, 1, gx - 1, 1, 0, 1, 0},
        {"row 1 (12 cores) -> (0,1), NOC1", 0, 1, gx - 1, 1, 0, 1, 1},
        {"col 0 (10 cores) -> (0,0), NOC0", 0, 0, 0, gy - 1, 0, 0, 0},
    };
    const char* mode_name[] = {"non-posted", "posted", "non-posted+barrier"};
    std::printf("# atomic stress, %u increments per member\n", k);
    for (const Case& c : cases) {
        for (uint32_t mode = 0; mode < 3; mode++) {
            uint64_t lost_total = 0;
            std::string detail;
            for (int r = 0; r < reps; r++) {
                tm::Program program = tm::CreateProgram();
                const tm::CoreRange rng({c.x0, c.y0}, {c.x1, c.y1});
                tm::CreateCircularBuffer(
                    program,
                    rng,
                    tm::CircularBufferConfig(
                        64, {{0, tt::DataFormat::UInt32}}
                    ).set_page_size(0, 64)
                );
                const uint32_t sem = tm::CreateSemaphore(program, rng, 0);
                std::vector<uint32_t> ct;
                tm::TensorAccessorArgs(*out).append_to(ct);
                const auto kid = tm::CreateKernel(
                    program,
                    kernel,
                    rng,
                    tm::DataMovementConfig{
                        .processor = tm::DataMovementProcessor::RISCV_0,
                        .noc =
                            c.noc ? tm::NOC::RISCV_1_default : tm::NOC::RISCV_0_default,
                        .compile_args = ct
                    }
                );
                const auto lead = tr(c.lx, c.ly);
                const uint32_t members =
                    (c.x1 - c.x0 + 1) * (c.y1 - c.y0 + 1) - 1 - c.noise;
                const uint32_t nx = 6, ny = gy - 1;
                const auto na = tr(0, 0), nb = tr(6, gy - 1);
                for (uint32_t y = c.y0; y <= c.y1; y++) {
                    for (uint32_t x = c.x0; x <= c.x1; x++) {
                        const bool noise = c.noise && x == nx && y == ny;
                        tm::SetRuntimeArgs(
                            program,
                            kid,
                            tm::CoreCoord{x, y},
                            {static_cast<uint32_t>(lead.x()),
                             static_cast<uint32_t>(lead.y()),
                             (x == c.lx && y == c.ly) ? 1u : 0u,
                             members,
                             k,
                             mode,
                             sem,
                             static_cast<uint32_t>(out->address()),
                             noise ? 1u : 0u,
                             static_cast<uint32_t>(na.x()),
                             static_cast<uint32_t>(na.y()),
                             static_cast<uint32_t>(nb.x()),
                             static_cast<uint32_t>(nb.y()),
                             7u * gy - 1u}
                        );
                    }
                }
                dist::MeshWorkload w;
                w.add_program(
                    dist::MeshCoordinateRange(mesh->shape()), std::move(program)
                );
                dist::EnqueueMeshWorkload(mesh->mesh_command_queue(), w, false);
                dist::Finish(mesh->mesh_command_queue());
                std::vector<uint32_t> res(8, 0);
                dist::EnqueueReadMeshBuffer(mesh->mesh_command_queue(), res, out, true);
                const int64_t lost =
                    static_cast<int64_t>(res[1]) - static_cast<int64_t>(res[0]);
                lost_total += lost > 0 ? lost : 0;
                char buf[96];
                std::snprintf(
                    buf,
                    sizeof(buf),
                    " [%u/%u%s, %.2f ticks/inc%s%u]",
                    res[0],
                    res[1],
                    res[3] ? " TIMEOUT" : "",
                    res[1] ? static_cast<double>(res[2]) / res[1] : 0.0,
                    c.noise ? ", noise word " : "",
                    c.noise ? res[4] : 0u
                );
                detail += buf;
            }
            std::printf(
                "%-34s %-19s lost=%llu%s\n",
                c.name,
                mode_name[mode],
                (unsigned long long)lost_total,
                detail.c_str()
            );
            std::fflush(stdout);
        }
    }
}

// Host <-> X280 LIM latency through persistent UMD IoWindows (no active X280 session,
// no kernels).
void hostio(X280Runtime& rt, const std::shared_ptr<MeshDevice>& mesh) {
    const auto ids = rt.tile_ids();
    X280Tile& tile = rt.tile(ids.front());
    const auto l2 = rt.virtual_core_from_tile(ids.front(), tt::tt_metal::NOC::NOC_0);
    const uint64_t base = tile.available_lim().begin + 0xC00;
    tile.write_available_lim(base, std::vector<uint8_t>(0x440, 0));
    auto& umd_cluster =
        *tt::tt_metal::MetalContext::instance().get_cluster().get_driver();
    auto* dev = umd_cluster.get_tt_device(mesh->get_devices().front()->id());
    using clk = std::chrono::steady_clock;

    struct V {
        const char* name;
        tt::umd::HostMemoryCaching caching;
        tt::umd::IoOrdering ordering;
    };

    const V variants[] = {
        {"UC strict", tt::umd::HostMemoryCaching::UC, tt::umd::IoOrdering::Strict},
        {"UC posted", tt::umd::HostMemoryCaching::UC, tt::umd::IoOrdering::Posted},
        {"WC strict", tt::umd::HostMemoryCaching::WC, tt::umd::IoOrdering::Strict},
        {"WC posted", tt::umd::HostMemoryCaching::WC, tt::umd::IoOrdering::Posted},
    };
    for (const auto& v : variants) {
        auto win = dev->create_io_window(
            tt::umd::TargetIoWindowConfig{
                .core_start =
                    {static_cast<uint32_t>(l2.x()), static_cast<uint32_t>(l2.y())},
                .addr = base,
                .noc = tt::umd::NocId::NOC0
            },
            tt::umd::HostIoWindowConfig{.mapping = v.caching},
            v.ordering
        );
        constexpr int kW = 20000, kR = 2000;
        auto t0 = clk::now();
        for (int i = 0; i < kW; i++) {
            win->write32((i % 16) * 64, i);
        }
        const double w_us =
            std::chrono::duration<double>(clk::now() - t0).count() * 1e6 / kW;
        uint32_t sink = 0;
        t0 = clk::now();
        for (int i = 0; i < kR; i++) {
            sink += win->read32(0x400);
        }
        const double r_us =
            std::chrono::duration<double>(clk::now() - t0).count() * 1e6 / kR;
        // Write-then-read-back round trip on the same word (visibility latency upper
        // bound).
        t0 = clk::now();
        for (int i = 0; i < kR; i++) {
            win->write32(0x400, i);
            while (win->read32(0x400) != static_cast<uint32_t>(i)) {
            }
        }
        const double rt_us =
            std::chrono::duration<double>(clk::now() - t0).count() * 1e6 / kR;
        std::printf(
            "%-10s write32 %.3f us  read32 %.3f us  write+readback %.3f us  (size %zu, "
            "sink %u)\n",
            v.name,
            w_us,
            r_us,
            rt_us,
            win->get_size(),
            sink & 1
        );
        std::fflush(stdout);
    }
}

}  // namespace

// Dispatches to the subcommand named by argv[1].
int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "compute";
    std::vector<uint32_t> plls;
    for (int i = 2; i < argc; i++) {
        plls.push_back(static_cast<uint32_t>(std::stoul(argv[i])));
    }
    if (plls.empty()) {
        plls = {0};
    }
    auto mesh = MeshDevice::create_unit_mesh(0);
    if (cmd == "atomic") {
        atomic(mesh);
        return 0;
    }
    auto opened = X280Runtime::open(
        mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared}
    );
    if (!opened.ready()) {
        std::fprintf(stderr, "X280 runtime not ready: %s\n", opened.diagnostic.c_str());
        return 2;
    }
    X280Runtime& rt = *opened.runtime;
    boot_all(rt);
    if (cmd == "compute") {
        compute(rt, plls);
    } else if (cmd == "coords") {
        const auto grid = mesh->compute_with_storage_grid_size();
        for (uint32_t y = 0; y < grid.y; y++) {
            std::printf("logical y=%u:", y);
            for (uint32_t x = 0; x < grid.x; x++) {
                const auto t = X280Runtime::virtual_core_from_logical_core(
                    *mesh,
                    tt::tt_metal::CoreCoord{x, y},
                    tt::CoreType::WORKER,
                    tt::tt_metal::NOC::NOC_0
                );
                std::printf(" (%u,%u)", t.x(), t.y());
            }
            std::printf("\n");
        }
        for (const auto& id : rt.tile_ids()) {
            const auto t = rt.virtual_core_from_tile(id, tt::tt_metal::NOC::NOC_0);
            std::printf(
                "l2cpu tile hw%d noc0 (%u,%u) -> translated (%u,%u)\n",
                id.hw_index,
                id.noc_x,
                id.noc_y,
                t.x(),
                t.y()
            );
        }
    } else if (cmd == "pingpong") {
        for (uint32_t mhz : plls) {
            if (mhz) {
                rt.set_pll(rt.tile_ids().front().chip_id, mhz);
            }
        }
        pingpong(rt, mesh);
    } else if (cmd == "sched") {
        for (uint32_t mhz : plls) {
            if (mhz) {
                rt.set_pll(rt.tile_ids().front().chip_id, mhz);
            }
        }
        sched(rt, mesh, 10000);
    } else if (cmd == "hostio") {
        hostio(rt, mesh);
    } else if (cmd == "disp") {
        for (uint32_t mhz : plls) {
            if (mhz) {
                rt.set_pll(rt.tile_ids().front().chip_id, mhz);
            }
        }
        disp(
            rt, mesh, std::getenv("TTX_N") ? std::stoull(std::getenv("TTX_N")) : 10000
        );
    } else {
        std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
        return 1;
    }
    return 0;
}
