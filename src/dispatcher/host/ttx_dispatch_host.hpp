// tt-x280-dispatch model v1 — host-side template (docs: docs/dispatch-model.md).
//   DispatchPlan plan = DispatchPlan::make(mesh, rt, depth);     // panels, groups, LIM
//   layout plan.add_ctl_cb(program, all_cores);                          // control
//   block CB (kCtlCb) on every core args.insert(args.end(), plan.worker_args(x, y)); //
//   14 args per data-movement kernel using it auto s = plan.start(n_cmds);
//   EnqueueMeshWorkload(...);  uint64_t ticks = plan.finish(s);  Finish(...);
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <immintrin.h>
#include <umd/device/chip_helpers/sysmem_buffer.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/io_window_config.hpp>
#include "x280/device/x280_runtime.hpp"

namespace ttx {

constexpr uint32_t kCtlCb = 8;
constexpr uint32_t kCtlBytes = 1024;
constexpr double kMtimeHz = 50e6;

struct DispatchPlan {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh;
    x280::X280Runtime* rt = nullptr;
    x280::X280Tile* tile = nullptr;
    uint32_t gx = 0, gy = 0, l2x = 0, l2y = 0;
    uint64_t args_addr = 0, mailbox = 0, acks = 0;
    std::vector<std::array<uint32_t, 2>>
        panels;  // logical column ranges, physically contiguous in NOC0
    uint32_t depth = 1, ring_stride = 16, tag = 1;
    uint64_t ordering = 2;  // posted
    uint64_t run = 0;
    uint32_t trace_addr =
        0;  // DRAM trace tensor (page per core, 16 B per command); 0 = tracing off

    // Host-fed streaming (start(n, true)): commands come from a 16-slot ring of 64 B
    // slots in LIM at hring, written through a persistent uncached PCIe window; the
    // X280 publishes its completed count at hring + kDoneOff.
    static constexpr uint32_t kHostDepth = 16;
    static constexpr uint64_t kHringOff = 0xC00;  // from args_addr
    static constexpr uint64_t kDoneOff = 0x400;   // from hring
    bool host_fed = false;
    std::shared_ptr<tt::umd::IoWindow>
        win;  // host -> LIM command ring (write-combining)
    std::shared_ptr<tt::umd::SysmemBuffer>
        done_buf;  // X280 -> pinned host memory completion count
    volatile uint32_t* done_va = nullptr;
    uint32_t pcie_x = 0, pcie_y = 0;
    uint64_t done_noc = 0;
    uint32_t done_seen = 0;

    std::array<uint32_t, 2> translated(uint32_t x, uint32_t y) const {
        const auto t = x280::X280Runtime::virtual_core_from_logical_core(
            *mesh,
            tt::tt_metal::CoreCoord{x, y},
            tt::CoreType::WORKER,
            tt::tt_metal::NOC::NOC_0
        );
        return {static_cast<uint32_t>(t.x()), static_cast<uint32_t>(t.y())};
    }

    static DispatchPlan make(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
        x280::X280Runtime& rt,
        uint32_t depth
    ) {
        DispatchPlan p;
        p.mesh = mesh;
        p.rt = &rt;
        p.depth = depth;
        const auto ids = rt.tile_ids();
        p.tile = &rt.tile(ids.front());
        const auto l2 =
            rt.virtual_core_from_tile(ids.front(), tt::tt_metal::NOC::NOC_0);
        p.l2x = static_cast<uint32_t>(l2.x());
        p.l2y = static_cast<uint32_t>(l2.y());
        p.args_addr = p.tile->available_lim().begin;
        p.mailbox = p.args_addr + 0x100;
        p.acks = p.args_addr + 0x400;
        const auto g = mesh->compute_with_storage_grid_size();
        p.gx = g.x;
        p.gy = g.y;
        const auto& soc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(
                mesh->get_devices().front()->id()
            );
        auto phys_x = [&](uint32_t lx) {
            const auto t = p.translated(lx, 0);
            return soc
                .translate_coord_to(
                    tt::umd::CoreCoord(
                        t[0], t[1], tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED
                    ),
                    tt::CoordSystem::NOC0
                )
                .x;
        };
        for (uint32_t x = 0; x < p.gx; x++) {
            if (!p.panels.empty() && phys_x(x) == phys_x(p.panels.back()[1]) + 1) {
                p.panels.back()[1] = x;
            } else {
                p.panels.push_back({x, x});
            }
        }
        if (p.panels.size() > 2) {
            throw std::runtime_error(
                "dispatch v1 supports at most two multicast panels"
            );
        }
        return p;
    }

    void add_ctl_cb(
        tt::tt_metal::Program& program, const tt::tt_metal::CoreRange& all
    ) const {
        tt::tt_metal::CreateCircularBuffer(
            program,
            all,
            tt::tt_metal::CircularBufferConfig(
                kCtlBytes, {{kCtlCb, tt::DataFormat::UInt32}}
            ).set_page_size(kCtlCb, kCtlBytes)
        );
    }

    // Column groups: leader = bottom core; rank = row; ack slot = column. Matches
    // ttx::Args field order.
    std::vector<uint32_t> worker_args(uint32_t x, uint32_t y) const {
        const auto gl = translated(x, gy - 1);
        const bool is_gl = y == gy - 1;
        return {
            l2x,
            l2y,
            static_cast<uint32_t>(mailbox),
            static_cast<uint32_t>(acks),
            (x == 0 && y == 0) ? 1u : 0u,
            gl[0],
            gl[1],
            gy - 1,
            is_gl ? 0u : y,
            x,
            is_gl ? 1u : 0u,
            tag,
            depth,
            ring_stride,
            trace_addr,
            y * gx + x
        };
    }

    // AICLK of the chip in MHz (the Tensix wall clock used by trace records).
    int aiclk_mhz() const {
        return tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(
            mesh->get_devices().front()->id()
        );
    }

    // Prepares LIM and starts ttx-sequencer.bin for `n_cmds` commands. Call before
    // EnqueueMeshWorkload; it picks a fresh tag, so call it before building worker_args
    // for the program.
    x280::SessionId start(uint64_t n_cmds, bool from_host = false) {
        static uint64_t run_base = 0x6b74640000000000ull ^
                                   (static_cast<uint64_t>(std::random_device{}()) << 8);
        run = ++run_base;
        tag = ((std::random_device{}() ^ static_cast<uint32_t>(run)) & 0xFFFFu) | 1u;
        host_fed = from_host;
        done_seen = 0;
        tile->write_available_lim(mailbox, std::vector<uint8_t>(64, 0));
        tile->write_available_lim(acks, std::vector<uint8_t>(2048, 0));
        const uint64_t hring = args_addr + kHringOff;
        if (host_fed) {
            tile->write_available_lim(hring, std::vector<uint8_t>(kDoneOff + 64, 0));
            if (!win) {
                const auto chip = mesh->get_devices().front()->id();
                auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
                auto* dev = cluster.get_driver()->get_tt_device(chip);
                win = dev->create_io_window(
                    tt::umd::TargetIoWindowConfig{
                        .core_start = {l2x, l2y},
                        .addr = hring,
                        .noc = tt::umd::NocId::NOC0
                    },
                    tt::umd::HostIoWindowConfig{
                        .mapping = tt::umd::HostMemoryCaching::WC
                    }
                );
                done_buf =
                    cluster.allocate_sysmem_buffer(chip, 4096, /*map_to_noc=*/true);
                done_va = static_cast<volatile uint32_t*>(done_buf->get_va());
                done_noc = done_buf->get_noc_address().value();
                const auto pcie = cluster.get_soc_desc(chip).get_cores(
                    tt::CoreType::PCIE, tt::CoordSystem::TRANSLATED
                );
                pcie_x = pcie.front().x;
                pcie_y = pcie.front().y;
            }
            done_va[0] = 0;
        }
        std::vector<uint8_t> block(192, 0);
        const uint64_t head[6] = {run, mailbox, acks, n_cmds, gx, panels.size()};
        uint32_t rect[2][4] = {};
        for (size_t r = 0; r < panels.size(); r++) {
            const auto a = translated(panels[r][0], 0),
                       b = translated(panels[r][1], gy - 1);
            rect[r][0] = a[0];
            rect[r][1] = a[1];
            rect[r][2] = b[0];
            rect[r][3] = b[1];
        }
        const uint64_t tail[13] = {
            1,
            depth,
            16,
            host_fed ? hring : 0,
            kHostDepth,
            host_fed ? hring + kDoneOff : 0,
            ordering,
            ring_stride,
            0,
            0,
            pcie_x,
            pcie_y,
            host_fed ? done_noc : 0
        };
        std::memcpy(block.data(), head, sizeof(head));
        std::memcpy(block.data() + sizeof(head), rect, sizeof(rect));
        std::memcpy(block.data() + sizeof(head) + sizeof(rect), tail, sizeof(tail));
        tile->write_available_lim(args_addr, block);
        x280::ActiveRunRequest req;
        req.image_path = std::string(X280_IMAGE_DIR) + "/ttx-sequencer.bin";
        req.run_value = run;
        req.args_addr = args_addr;
        req.args_bytes = block.size();
        const auto enq = rt->enqueue(*tile, req);
        if (!enq.session) {
            throw std::runtime_error("X280 enqueue failed: " + x280::describe(enq));
        }
        return enq.session;
    }

    // Host-fed only: completed command count, written by the X280 into pinned host
    // memory (local read).
    uint32_t completed() {
        done_seen = done_va[0];
        return done_seen;
    }

    // Host-fed only: enqueue command `seq` (seq = 1, 2, ... in order). One posted MMIO
    // store; polls the completed count only when the ring would overrun.
    void submit(uint32_t seq) {
        while (seq - done_seen >= kHostDepth) {
            completed();
        }
        win->write32((seq % kHostDepth) * 64, seq);
        _mm_sfence();  // flush the write-combining buffer so the command leaves the
                       // host now
    }

    // Host-fed only: block until command `seq` has completed on every core.
    void wait(uint32_t seq) {
        while (static_cast<int32_t>(done_seen - seq) < 0) {
            completed();
        }
    }

    // Waits for the X280 image, cleans up, and returns mtime ticks from command 1 to
    // the last completion (0 on failure).
    uint64_t finish(
        x280::SessionId s, std::chrono::seconds timeout = std::chrono::seconds(60)
    ) {
        const auto res = rt->session(s).wait_until_result_ready(timeout);
        const auto cleaned = rt->cleanup(s, std::chrono::seconds(5));
        if (!res.succeeded() || !cleaned.cleaned_up()) {
            return 0;
        }
        const uint64_t ticks = res.hartstates[0].state.result;
        return ticks >= ~16ull ? 0 : ticks;
    }
};

}  // namespace ttx
