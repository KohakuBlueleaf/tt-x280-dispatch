// ttx_mlp [pcM,pcN,Kt[,bw] ...]: X280-dispatched MLP forward Y = silu(X W1) W2 on
// the full Tensix grid (P9b). X: M x K, W1: K x F, W2: F x F, M = gy * pcM tiles, F =
// gx * pcN tiles. One X280 command per op, depth 1 (dependent). env: TTX_REPS (forward
// passes, default 64), TTX_PLL (default 1750), TTX_FID (HiFi2), TTX_CSV (append
// results).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttx_dispatch_host.hpp"

namespace mtl = tt::tt_metal;
namespace dist = tt::tt_metal::distributed;
using x280::X280Runtime;

namespace {

const std::string kKernels = TTX_KERNEL_DIR;

struct Shape {
    uint32_t pcM, pcN, Kt, bw;
};

// Output subblock (h, w) dividing the core block, h * w <= 4.
std::pair<uint32_t, uint32_t> pick_subblock(uint32_t pcM, uint32_t pcN) {
    static const uint32_t choices[][2] = {
        {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1}
    };
    for (const auto& c : choices) {
        if (pcM % c[0] == 0 && pcN % c[1] == 0) {
            return {c[0], c[1]};
        }
    }
    return {1, 1};
}

// SiLU in float, for the reference.
float silu_ref(float v) {
    return v / (1.0f + std::exp(-v));
}

// One MLP shape: resident run (X280-issued or host-fed), timed and checked.
void run_shape(
    const std::shared_ptr<dist::MeshDevice>& mesh,
    X280Runtime& rt,
    const Shape& s,
    uint32_t reps,
    mtl::MathFidelity fid,
    uint32_t pll,
    FILE* csv
) {
    auto plan = ttx::DispatchPlan::make(mesh, rt, 1);
    const uint32_t gx = plan.gx, gy = plan.gy;
    const uint32_t Mt = gy * s.pcM, Ft = gx * s.pcN, Kt = s.Kt;
    const uint32_t M = Mt * 32, F = Ft * 32, K = Kt * 32;
    const double flop = 2.0 * M * K * F + 2.0 * M * F * F;
    std::printf(
        "## MLP X[%u x %u] W1[%u x %u] W2[%u x %u] (per core %ux%u tiles, bw %u)  %.2f "
        "GFLOP/forward\n",
        M,
        K,
        K,
        F,
        F,
        F,
        s.pcM,
        s.pcN,
        s.bw,
        flop / 1e9
    );
    if (Kt % s.bw || Ft % s.bw) {
        std::printf("   skip: K and F tiles must be multiples of bw\n");
        return;
    }
    std::mt19937 rng(77 + M + F + K);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<bfloat16> X(size_t(M) * K), W1(size_t(K) * F), W2(size_t(F) * F);
    for (auto& v : X) {
        v = bfloat16(U(rng));
    }
    for (auto& v : W1) {
        v = bfloat16(U(rng) * 0.1f);
    }
    for (auto& v : W2) {
        v = bfloat16(U(rng) * 0.1f);
    }

    const uint32_t tile = 2048;
    dist::DeviceLocalBufferConfig dram{
        .page_size = tile, .buffer_type = mtl::BufferType::DRAM
    };
    auto mk = [&](uint64_t tiles) {
        return dist::MeshBuffer::create(
            dist::ReplicatedBufferConfig{.size = tiles * tile}, dram, mesh.get()
        );
    };
    auto bX = mk(uint64_t(Mt) * Kt), bW1 = mk(uint64_t(Kt) * Ft),
         bH = mk(uint64_t(Mt) * Ft), bH2 = mk(uint64_t(Mt) * Ft),
         bW2 = mk(uint64_t(Ft) * Ft), bY = mk(uint64_t(Mt) * Ft);
    auto& cq = mesh->mesh_command_queue();
    auto tX = tilize_nfaces(X, M, K), tW1 = tilize_nfaces(W1, K, F),
         tW2 = tilize_nfaces(W2, F, F);
    dist::EnqueueWriteMeshBuffer(cq, bX, tX, true);
    dist::EnqueueWriteMeshBuffer(cq, bW1, tW1, true);
    dist::EnqueueWriteMeshBuffer(cq, bW2, tW2, true);

    const auto [sbh, sbw] = pick_subblock(s.pcM, s.pcN);
    const uint32_t n_cmds = 3 * reps;
    // TTX_HOSTFED=1: host enqueues each op through the mapped LIM ring (streaming, then
    // synchronous forwards).
    const bool host_fed =
        std::getenv("TTX_HOSTFED") && std::atoi(std::getenv("TTX_HOSTFED"));
    for (int rep = 0; rep < 2; rep++) {  // rep 0 compiles and warms
        const auto session = plan.start(n_cmds, host_fed);

        mtl::Program p = mtl::CreateProgram();
        const mtl::CoreRange all({0, 0}, {gx - 1, gy - 1});
        const auto df = tt::DataFormat::Float16_b;
        mtl::CreateCircularBuffer(
            p,
            all,
            mtl::CircularBufferConfig(
                2 * s.pcM * s.bw * tile, {{0, df}}
            ).set_page_size(0, tile)
        );
        mtl::CreateCircularBuffer(
            p,
            all,
            mtl::CircularBufferConfig(
                2 * s.pcN * s.bw * tile, {{1, df}}
            ).set_page_size(1, tile)
        );
        mtl::CreateCircularBuffer(
            p,
            all,
            mtl::CircularBufferConfig(2 * tile, {{2, df}}).set_page_size(2, tile)
        );
        mtl::CreateCircularBuffer(
            p,
            all,
            mtl::CircularBufferConfig(s.pcM * s.pcN * tile, {{16, df}, {24, df}})
                .set_page_size(16, tile)
                .set_page_size(24, tile)
        );
        plan.add_ctl_cb(p, all);
        std::vector<uint32_t> acc_ct;
        mtl::TensorAccessorArgs(*bX).append_to(acc_ct);

        std::vector<mtl::CoreRange> roles[2][2], wr_noc0, wr_noc1;
        for (const auto& pn : plan.panels) {
            roles[1][1].push_back(mtl::CoreRange({pn[0], 0}, {pn[0], 0}));
            roles[1][0].push_back(mtl::CoreRange({pn[0], 1}, {pn[0], gy - 1}));
            wr_noc1.push_back(mtl::CoreRange({pn[0], 0}, {pn[0], gy - 1}));
            if (pn[1] > pn[0]) {
                roles[0][1].push_back(mtl::CoreRange({pn[0] + 1, 0}, {pn[1], 0}));
                roles[0][0].push_back(mtl::CoreRange({pn[0] + 1, 1}, {pn[1], gy - 1}));
                wr_noc1.push_back(mtl::CoreRange({pn[0] + 1, 0}, {pn[1], 0}));
                wr_noc0.push_back(mtl::CoreRange({pn[0] + 1, 1}, {pn[1], gy - 1}));
            }
        }
        mtl::KernelHandle rk[2][2];
        for (int i0 = 0; i0 < 2; i0++) {
            for (int i1 = 0; i1 < 2; i1++) {
                std::map<std::string, std::string> defs;
                if (i0) {
                    defs["IN0_SENDER"] = "1";
                }
                if (i1) {
                    defs["IN1_SENDER"] = "1";
                }
                rk[i0][i1] = mtl::CreateKernel(
                    p,
                    kKernels + "/mlp/reader.cpp",
                    mtl::CoreRangeSet(roles[i0][i1]),
                    mtl::DataMovementConfig{
                        .processor = mtl::DataMovementProcessor::RISCV_1,
                        .noc = (i0 || i1) ? mtl::NOC::RISCV_0_default
                                          : mtl::NOC::RISCV_1_default,
                        .compile_args = acc_ct,
                        .defines = defs
                    }
                );
            }
        }
        const auto wk1 = mtl::CreateKernel(
            p,
            kKernels + "/mlp/writer.cpp",
            mtl::CoreRangeSet(wr_noc1),
            mtl::DataMovementConfig{
                .processor = mtl::DataMovementProcessor::RISCV_0,
                .noc = mtl::NOC::RISCV_1_default,
                .compile_args = acc_ct
            }
        );
        const auto wk0 = mtl::CreateKernel(
            p,
            kKernels + "/mlp/writer.cpp",
            mtl::CoreRangeSet(wr_noc0),
            mtl::DataMovementConfig{
                .processor = mtl::DataMovementProcessor::RISCV_0,
                .noc = mtl::NOC::RISCV_0_default,
                .compile_args = acc_ct
            }
        );
        const std::vector<uint32_t> cargs = {
            s.bw,
            s.pcM / sbh,
            s.pcM * s.bw,
            sbh * s.bw,
            s.pcN / sbw,
            s.pcN * s.bw,
            s.pcN,
            Kt / s.bw,
            Ft / s.bw,
            sbh,
            sbw,
            sbh * sbw,
            n_cmds,
            s.pcM * s.pcN
        };
        mtl::CreateKernel(
            p,
            kKernels + "/mlp/compute.cpp",
            all,
            mtl::ComputeConfig{
                .math_fidelity = fid, .fp32_dest_acc_en = true, .compile_args = cargs
            }
        );
        const uint32_t sem[4] = {
            mtl::CreateSemaphore(p, all, 0),
            mtl::CreateSemaphore(p, all, 0),
            mtl::CreateSemaphore(p, all, 0),
            mtl::CreateSemaphore(p, all, 0)
        };
        auto A = [](const std::shared_ptr<dist::MeshBuffer>& b) {
            return static_cast<uint32_t>(b->address());
        };
        for (const auto& pn : plan.panels) {
            for (uint32_t y = 0; y < gy; y++) {
                for (uint32_t x = pn[0]; x <= pn[1]; x++) {
                    const bool s0 = x == pn[0], s1 = y == 0;
                    const auto left = plan.translated(pn[0], y),
                               left1 = plan.translated(std::min(pn[0] + 1, pn[1]), y),
                               right = plan.translated(pn[1], y);
                    const auto top = plan.translated(x, 0),
                               top1 = plan.translated(x, 1),
                               bottom = plan.translated(x, gy - 1);
                    std::vector<uint32_t> ra = {
                        A(bX),
                        Kt * s.pcM * y,
                        Kt,
                        A(bW1),
                        s.pcN * x,
                        Ft,
                        s.bw * Ft,
                        Kt / s.bw,
                        A(bH2),
                        Ft * s.pcM * y,
                        Ft,
                        A(bW2),
                        s.pcN * x,
                        Ft,
                        s.bw * Ft,
                        Ft / s.bw,
                        s.bw,
                        s.pcM,
                        s.pcN,
                        A(bH),
                        s.pcM * y * Ft + s.pcN * x,
                        Ft,
                        sbh,
                        sbw,
                        right[0],
                        right[1],
                        left1[0],
                        left1[1],
                        pn[1] - pn[0],
                        left[0],
                        left[1],
                        sem[0],
                        sem[1],
                        bottom[0],
                        bottom[1],
                        top1[0],
                        top1[1],
                        gy - 1,
                        top[0],
                        top[1],
                        sem[2],
                        sem[3],
                        n_cmds
                    };
                    const auto da = plan.worker_args(x, y);
                    ra.insert(ra.end(), da.begin(), da.end());
                    mtl::SetRuntimeArgs(p, rk[s0][s1], mtl::CoreCoord{x, y}, ra);
                    std::vector<uint32_t> wa = {
                        A(bH),
                        A(bH2),
                        A(bY),
                        x * s.pcN + y * s.pcM * Ft,
                        Ft,
                        sbw,
                        sbh,
                        s.pcN / sbw,
                        s.pcM / sbh,
                        n_cmds
                    };
                    wa.insert(wa.end(), da.begin(), da.end());
                    mtl::SetRuntimeArgs(
                        p, (s0 || s1) ? wk1 : wk0, mtl::CoreCoord{x, y}, wa
                    );
                }
            }
        }
        dist::MeshWorkload w;
        w.add_program(dist::MeshCoordinateRange(mesh->shape()), std::move(p));
        dist::EnqueueMeshWorkload(cq, w, false);
        double host_submit_us = 0, stream_us = 0, sync_us = 0;
        if (host_fed) {
            // First half: stream reps/2 forwards without waiting; second half: wait
            // after every forward.
            using clk = std::chrono::steady_clock;
            const uint32_t half = reps / 2;
            uint32_t seq = 0;
            double submit_s = 0;
            const auto t0 = clk::now();
            for (uint32_t f = 0; f < half; f++) {
                for (int op = 0; op < 3; op++) {
                    const auto s0 = clk::now();
                    plan.submit(++seq);
                    submit_s += std::chrono::duration<double>(clk::now() - s0).count();
                }
            }
            plan.wait(seq);
            const auto t1 = clk::now();
            for (uint32_t f = half; f < reps; f++) {
                for (int op = 0; op < 3; op++) {
                    plan.submit(++seq);
                }
                plan.wait(seq);
            }
            const auto t2 = clk::now();
            host_submit_us = submit_s * 1e6 / (3.0 * half);
            stream_us = std::chrono::duration<double>(t1 - t0).count() * 1e6 / half;
            sync_us =
                std::chrono::duration<double>(t2 - t1).count() * 1e6 / (reps - half);
        }
        const uint64_t ticks = plan.finish(session);
        dist::Finish(cq);
        if (ticks == 0) {
            std::printf("   X280 run FAILED (timeout)\n");
            return;
        }
        if (rep == 0) {
            continue;
        }
        const double us = ticks / ttx::kMtimeHz * 1e6 / reps;
        // Reference on 16 sampled rows: H and H2 rounded to bf16 as on device.
        std::vector<bfloat16> Y(size_t(M) * F);
        dist::EnqueueReadMeshBuffer(cq, Y, bY, true);
        Y = untilize_nfaces(Y, M, F);
        double se = 0, sr = 0, maxe = 0;
        std::vector<float> h2(F);
        for (int t = 0; t < 16; t++) {
            const uint32_t i = (t * 2654435761u) % M;
            for (uint32_t f = 0; f < F; f++) {
                double acc = 0;
                for (uint32_t k = 0; k < K; k++) {
                    acc += double(static_cast<float>(X[size_t(i) * K + k])) *
                           static_cast<float>(W1[size_t(k) * F + f]);
                }
                const float h = static_cast<float>(bfloat16(static_cast<float>(acc)));
                h2[f] = static_cast<float>(bfloat16(silu_ref(h)));
            }
            for (uint32_t j = 0; j < F; j++) {
                double acc = 0;
                for (uint32_t f = 0; f < F; f++) {
                    acc += double(h2[f]) * static_cast<float>(W2[size_t(f) * F + j]);
                }
                const double d = double(static_cast<float>(Y[size_t(i) * F + j])) - acc;
                se += d * d;
                sr += acc * acc;
                maxe = std::max(maxe, std::fabs(d));
            }
        }
        const double rel = std::sqrt(se / sr);
        std::printf(
            "   X280 MLP forward (3 dependent ops): %.2f us/forward (%.2f us/op)  %.1f "
            "TFLOPS  rel-rms %.2e  %s\n",
            us,
            us / 3,
            flop / (us * 1e-6) / 1e12,
            rel,
            rel < 2e-2 ? "OK" : "MISMATCH"
        );
        if (host_fed) {
            std::printf(
                "   host-fed: %.3f us host time per enqueued op; streaming %.2f "
                "us/forward; "
                "synchronous (wait each forward) %.2f us/forward\n",
                host_submit_us,
                stream_us,
                sync_us
            );
            if (csv) {
                std::fprintf(
                    csv,
                    "mlp_hostfed,%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f\n",
                    M,
                    K,
                    F,
                    s.bw,
                    pll,
                    reps,
                    host_submit_us,
                    stream_us,
                    sync_us
                );
                std::fflush(csv);
            }
        }
        if (csv) {
            std::fprintf(
                csv,
                "mlp,%u,%u,%u,%u,%u,%u,%.3f,%.2f,%.3e\n",
                M,
                K,
                F,
                s.bw,
                pll,
                reps,
                us,
                flop / (us * 1e-6) / 1e12,
                rel
            );
            std::fflush(csv);
        }
    }
}

}  // namespace

// Runs the MLP shapes given as arguments (or the defaults).
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<Shape> shapes;
    for (int i = 1; i < argc; i++) {
        Shape s{1, 1, 8, 2};
        std::sscanf(argv[i], "%u,%u,%u,%u", &s.pcM, &s.pcN, &s.Kt, &s.bw);
        shapes.push_back(s);
    }
    if (shapes.empty()) {
        shapes = {{1, 1, 8, 4}, {2, 2, 32, 8}, {4, 4, 64, 8}};
    }
    auto env = [](const char* k, uint32_t d) {
        return std::getenv(k) ? uint32_t(std::atoi(std::getenv(k))) : d;
    };
    const uint32_t reps = env("TTX_REPS", 64), pll = env("TTX_PLL", 1750);
    const std::string fids = std::getenv("TTX_FID") ? std::getenv("TTX_FID") : "HiFi2";
    const auto fid = fids == "LoFi"    ? mtl::MathFidelity::LoFi
                     : fids == "HiFi4" ? mtl::MathFidelity::HiFi4
                                       : mtl::MathFidelity::HiFi2;
    FILE* csv =
        std::getenv("TTX_CSV") ? std::fopen(std::getenv("TTX_CSV"), "a") : nullptr;

    auto mesh = dist::MeshDevice::create_unit_mesh(0);
    auto opened = X280Runtime::open(
        mesh,
        x280::RuntimeOpenConfig{
            .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
        }
    );
    if (!opened.ready()) {
        std::fprintf(stderr, "X280 runtime not ready: %s\n", opened.diagnostic.c_str());
        return 2;
    }
    X280Runtime& rt = *opened.runtime;
    bool all_idle = true;
    for (const auto& id : rt.tile_ids()) {
        all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
    }
    if (!all_idle) {
        x280::MultiTileBootRequest req;
        req.tiles = rt.tile_ids();
        req.idle_image_path = std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin";
        req.allow_reset_release = true;
        if (!rt.boot_tiles(req).ready()) {
            std::fprintf(stderr, "boot_tiles failed\n");
            return 2;
        }
    }
    rt.set_pll(rt.tile_ids().front().chip_id, pll);
    std::printf(
        "# X280 dispatch v1, PLL %u MHz, fidelity %s, fp32 acc, %u forwards\n",
        pll,
        fids.c_str(),
        reps
    );
    for (const auto& s : shapes) {
        run_shape(mesh, rt, s, reps, fid, pll, csv);
    }
    if (csv) {
        std::fclose(csv);
    }
    return 0;
}
