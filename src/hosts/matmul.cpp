// ttx_matmul [pcM,pcN,Kt[,bw] ...]: X280-sequenced persistent matmul on the full
// Tensix grid. env: TTX_N (default 64), TTX_FID (LoFi|HiFi2|HiFi3|HiFi4),
// TTX_PLL (default 1750), TTX_VARIANTS (subset of "xsr").
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "x280/device/x280_runtime.hpp"
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

using namespace x280;
namespace mtl = tt::tt_metal;
namespace dist = tt::tt_metal::distributed;
using dist::MeshDevice;

namespace {

constexpr double kMtimeHz = 50e6;
const std::string kImageDir = X280_IMAGE_DIR;
const std::string kKernels = TTX_KERNEL_DIR;
uint64_t g_run = 0x6b746d0000000000ull;

struct Shape {
    uint32_t pcM, pcN, Kt, bw;
};

struct Panel {
    uint32_t x0, x1;  // logical columns, inclusive
};

struct X280Ctl {
    uint32_t l2x, l2y, mailbox, acks;
    uint32_t group =
        0;  // 0 flat acks, 1 column groups (leader = bottom core, write-gather)
    uint32_t depth = 1;  // command ring depth (<= 4)
    uint32_t rstride = 16;
    uint32_t tag = 0;
};

struct Env {
    std::shared_ptr<MeshDevice> mesh;
    uint32_t gx, gy;
    std::vector<Panel> panels;
    mtl::MathFidelity fid;
    bool fp32acc = false;
};

struct XY {
    uint32_t x, y;
};

// Translated NOC0 coordinates of logical core (x, y).
XY tr(const Env& e, uint32_t x, uint32_t y) {
    const auto t = X280Runtime::virtual_core_from_logical_core(
        *e.mesh, mtl::CoreCoord{x, y}, tt::CoreType::WORKER, mtl::NOC::NOC_0
    );
    return {uint32_t(t.x()), uint32_t(t.y())};
}

// Output subblock (h, w) dividing the core block, h * w <= 4.
std::pair<uint32_t, uint32_t> pick_subblock(
    uint32_t pcM, uint32_t pcN, uint32_t max_tiles
) {
    static const uint32_t choices[][2] = {{4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1},
                                          {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
                                          {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4},
                                          {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1}};
    for (const auto& c : choices) {
        if (c[0] * c[1] <= max_tiles && pcM % c[0] == 0 && pcN % c[1] == 0) {
            return {c[0], c[1]};
        }
    }
    return {1, 1};
}

struct Buffers {
    std::shared_ptr<dist::MeshBuffer> a, b, c;
};

// Builds the persistent matmul program over all panels. standalone: no X280 protocol.
mtl::Program build(
    const Env& e,
    const Shape& s,
    const Buffers& buf,
    uint32_t n,
    bool standalone,
    const X280Ctl& x
) {
    const uint32_t Mt = e.gy * s.pcM, Nt = e.gx * s.pcN, Kt = s.Kt;
    (void)Mt;
    const auto df = tt::DataFormat::Float16_b;
    const uint32_t tile = tt::tile_size(df);
    const auto [sbh, sbw] = pick_subblock(s.pcM, s.pcN, e.fp32acc ? 4 : 8);
    const uint32_t num_blocks = Kt / s.bw;

    mtl::Program p = mtl::CreateProgram();
    const mtl::CoreRange all({0, 0}, {e.gx - 1, e.gy - 1});
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
        mtl::CircularBufferConfig(s.pcM * s.pcN * tile, {{16, df}, {24, df}})
            .set_page_size(16, tile)
            .set_page_size(24, tile)
    );
    mtl::CreateCircularBuffer(
        p,
        all,
        mtl::CircularBufferConfig(
            1024, {{8, tt::DataFormat::UInt32}}
        ).set_page_size(8, 1024)
    );

    std::vector<uint32_t> rd_ct, wr_ct;
    mtl::TensorAccessorArgs(*buf.a).append_to(rd_ct);
    mtl::TensorAccessorArgs(*buf.b).append_to(rd_ct);
    mtl::TensorAccessorArgs(*buf.c).append_to(wr_ct);

    // Role sets: [in0 sender][in1 sender]; panel-left column writers on NOC1.
    std::vector<mtl::CoreRange> roles[2][2];
    std::vector<mtl::CoreRange> wr_noc1, wr_noc0;
    for (const auto& pn : e.panels) {
        roles[1][1].push_back(mtl::CoreRange({pn.x0, 0}, {pn.x0, 0}));
        roles[1][0].push_back(mtl::CoreRange({pn.x0, 1}, {pn.x0, e.gy - 1}));
        if (pn.x1 > pn.x0) {
            roles[0][1].push_back(mtl::CoreRange({pn.x0 + 1, 0}, {pn.x1, 0}));
            roles[0][0].push_back(mtl::CoreRange({pn.x0 + 1, 1}, {pn.x1, e.gy - 1}));
            wr_noc0.push_back(mtl::CoreRange({pn.x0 + 1, 1}, {pn.x1, e.gy - 1}));
            wr_noc1.push_back(mtl::CoreRange({pn.x0 + 1, 0}, {pn.x1, 0}));
        }
        wr_noc1.push_back(mtl::CoreRange({pn.x0, 0}, {pn.x0, e.gy - 1}));
    }
    std::map<std::string, std::string> base_defs;
    if (standalone) {
        base_defs["TTX_STANDALONE"] = "1";
    }
    mtl::KernelHandle rk[2][2];
    for (int i0 = 0; i0 < 2; i0++) {
        for (int i1 = 0; i1 < 2; i1++) {
            auto defs = base_defs;
            if (i0) {
                defs["IN0_SENDER"] = "1";
            }
            if (i1) {
                defs["IN1_SENDER"] = "1";
            }
            // Multicasting readers on NOC0 (corner order start < end); pure receivers
            // on NOC1.
            rk[i0][i1] = mtl::CreateKernel(
                p,
                kKernels + "/matmul/reader.cpp",
                mtl::CoreRangeSet(roles[i0][i1]),
                mtl::DataMovementConfig{
                    .processor = mtl::DataMovementProcessor::RISCV_1,
                    .noc = (i0 || i1) ? mtl::NOC::RISCV_0_default
                                      : mtl::NOC::RISCV_1_default,
                    .compile_args = rd_ct,
                    .defines = defs
                }
            );
        }
    }
    const auto wk1 = mtl::CreateKernel(
        p,
        kKernels + "/matmul/writer.cpp",
        mtl::CoreRangeSet(wr_noc1),
        mtl::DataMovementConfig{
            .processor = mtl::DataMovementProcessor::RISCV_0,
            .noc = mtl::NOC::RISCV_1_default,
            .compile_args = wr_ct,
            .defines = base_defs
        }
    );
    mtl::KernelHandle wk0 = 0;
    if (!wr_noc0.empty()) {
        wk0 = mtl::CreateKernel(
            p,
            kKernels + "/matmul/writer.cpp",
            mtl::CoreRangeSet(wr_noc0),
            mtl::DataMovementConfig{
                .processor = mtl::DataMovementProcessor::RISCV_0,
                .noc = mtl::NOC::RISCV_0_default,
                .compile_args = wr_ct,
                .defines = base_defs
            }
        );
    }
    const uint32_t in0_sub = sbh * s.bw;
    std::vector<uint32_t> cargs = {
        s.bw,
        s.pcM / sbh,
        s.pcM * s.bw,
        in0_sub,
        s.pcN / sbw,
        s.pcN * s.bw,
        s.pcN,
        num_blocks,
        sbh,
        sbw,
        sbh * sbw,
        n
    };
    mtl::CreateKernel(
        p,
        kKernels + "/matmul/compute.cpp",
        all,
        mtl::ComputeConfig{
            .math_fidelity = e.fid, .fp32_dest_acc_en = e.fp32acc, .compile_args = cargs
        }
    );
    const uint32_t sem[4] = {
        mtl::CreateSemaphore(p, all, 0),
        mtl::CreateSemaphore(p, all, 0),
        mtl::CreateSemaphore(p, all, 0),
        mtl::CreateSemaphore(p, all, 0)
    };

    uint32_t idx = 0;
    for (const auto& pn : e.panels) {
        for (uint32_t y = 0; y < e.gy; y++) {
            for (uint32_t gxx = pn.x0; gxx <= pn.x1; gxx++, idx++) {
                const bool s0 = gxx == pn.x0, s1 = y == 0;
                const auto left = tr(e, pn.x0, y),
                           left1 = tr(e, std::min(pn.x0 + 1, pn.x1), y),
                           right = tr(e, pn.x1, y);
                const auto top = tr(e, gxx, 0), top1 = tr(e, gxx, 1),
                           bottom = tr(e, gxx, e.gy - 1);
                std::vector<uint32_t> ra = {
                    uint32_t(buf.a->address()),
                    Kt * s.pcM * y,
                    Kt,
                    s.bw,
                    s.bw,
                    s.pcM,
                    s.pcM * s.bw,
                    uint32_t(buf.b->address()),
                    s.pcN * gxx,
                    Nt,
                    s.bw * Nt,
                    s.pcN,
                    s.bw,
                    s.pcN * s.bw,
                    num_blocks,
                    right.x,
                    right.y,
                    left1.x,
                    left1.y,
                    pn.x1 - pn.x0,
                    left.x,
                    left.y,
                    sem[0],
                    sem[1],
                    bottom.x,
                    bottom.y,
                    top1.x,
                    top1.y,
                    e.gy - 1,
                    top.x,
                    top.y,
                    sem[2],
                    sem[3],
                    n,
                    x.l2x,
                    x.l2y,
                    x.mailbox,
                    x.acks,
                    idx,
                    idx == 0 ? 1u : 0u
                };
                // Column group of (gxx, y): leader = bottom core, rank = y, ack slot =
                // column.
                const auto gl = tr(e, gxx, e.gy - 1);
                const bool is_gl = y == e.gy - 1;
                const std::vector<uint32_t> grp = {
                    gl.x,
                    gl.y,
                    x.group ? e.gy - 1 : 0u,
                    is_gl ? 0u : y,
                    gxx,
                    is_gl ? 1u : 0u,
                    x.tag
                };
                ra.insert(ra.end(), grp.begin(), grp.end());
                ra.push_back(x.depth);
                ra.push_back(x.rstride);
                const mtl::CoreCoord core{gxx, y};
                mtl::SetRuntimeArgs(p, rk[s0][s1], core, ra);
                std::vector<uint32_t> wa = {
                    uint32_t(buf.c->address()),
                    gxx * s.pcN + y * s.pcM * Nt,
                    Nt,
                    sbw,
                    sbh * Nt,
                    sbw,
                    sbh,
                    sbw * sbh,
                    s.pcN / sbw,
                    s.pcM / sbh,
                    n,
                    x.l2x,
                    x.l2y,
                    x.acks,
                    idx
                };
                wa.insert(wa.end(), grp.begin(), grp.end());
                mtl::SetRuntimeArgs(p, (s0 || s1) ? wk1 : wk0, core, wa);
            }
        }
    }
    return p;
}

// Enqueues a program on the mesh, waits for it and returns the wall time in seconds.
double run_program(const Env& e, mtl::Program&& p) {
    dist::MeshWorkload w;
    w.add_program(dist::MeshCoordinateRange(e.mesh->shape()), std::move(p));
    const auto t0 = std::chrono::steady_clock::now();
    dist::EnqueueMeshWorkload(e.mesh->mesh_command_queue(), w, false);
    dist::Finish(e.mesh->mesh_command_queue());
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// One matmul shape: every variant, timed and checked against fp64.
void bench_shape(
    Env& e, X280Runtime* rtp, const Shape& s, uint32_t n, const std::string& variants
) {
    const uint32_t Mt = e.gy * s.pcM, Nt = e.gx * s.pcN, Kt = s.Kt;
    const uint32_t M = Mt * 32, N = Nt * 32, K = Kt * 32;
    const double flop = 2.0 * M * N * K;
    const uint32_t tile = 2048;
    std::printf(
        "## M=%u N=%u K=%u  (per core %ux%u tiles, Kt=%u, block_w=%u)  %.2f GFLOP\n",
        M,
        N,
        K,
        s.pcM,
        s.pcN,
        Kt,
        s.bw,
        flop / 1e9
    );
    if (Kt % s.bw) {
        std::printf("   skip: Kt %% bw != 0\n");
        return;
    }
    std::mt19937 rng(1234 + M + N + K);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<bfloat16> A(size_t(M) * K), B(size_t(K) * N);
    for (auto& v : A) {
        v = bfloat16(U(rng));
    }
    for (auto& v : B) {
        v = bfloat16(U(rng));
    }

    dist::DeviceLocalBufferConfig dram{
        .page_size = tile, .buffer_type = mtl::BufferType::DRAM
    };
    Buffers buf;
    buf.a = dist::MeshBuffer::create(
        dist::ReplicatedBufferConfig{.size = uint64_t(tile) * Mt * Kt},
        dram,
        e.mesh.get()
    );
    buf.b = dist::MeshBuffer::create(
        dist::ReplicatedBufferConfig{.size = uint64_t(tile) * Kt * Nt},
        dram,
        e.mesh.get()
    );
    buf.c = dist::MeshBuffer::create(
        dist::ReplicatedBufferConfig{.size = uint64_t(tile) * Mt * Nt},
        dram,
        e.mesh.get()
    );
    auto& cq = e.mesh->mesh_command_queue();
    auto At = tilize_nfaces(A, M, K);
    auto Bt = tilize_nfaces(B, K, N);
    dist::EnqueueWriteMeshBuffer(cq, buf.a, At, true);
    dist::EnqueueWriteMeshBuffer(cq, buf.b, Bt, true);

    auto check = [&](const char* tag) {
        std::vector<bfloat16> C(size_t(M) * N);
        dist::EnqueueReadMeshBuffer(cq, C, buf.c, true);
        C = untilize_nfaces(C, M, N);
        std::mt19937 r2(7);
        double se = 0, sr = 0, maxe = 0;
        for (int t = 0; t < 2000; t++) {
            const uint32_t i = r2() % M, j = r2() % N;
            double ref = 0;
            for (uint32_t k = 0; k < K; k++) {
                ref += double(static_cast<float>(A[size_t(i) * K + k])) *
                       double(static_cast<float>(B[size_t(k) * N + j]));
            }
            const double d = double(static_cast<float>(C[size_t(i) * N + j])) - ref;
            se += d * d;
            sr += ref * ref;
            maxe = std::max(maxe, std::fabs(d));
        }
        std::printf(
            "   [%s] check 2000 samples: rel-rms err %.2e, max abs err %.3f (|ref| rms "
            "%.2f)\n",
            tag,
            std::sqrt(se / sr),
            maxe,
            std::sqrt(sr / 2000)
        );
    };

    const X280Ctl none{0, 0, 0, 0};
    double x_per = 0;
    if (variants.find('x') != std::string::npos) {
        X280Runtime& rt = *rtp;
        const auto ids = rt.tile_ids();
        X280Tile& xt = rt.tile(ids.front());
        const auto l2 = rt.virtual_core_from_tile(ids.front(), mtl::NOC::NOC_0);
        const uint64_t args_addr = xt.available_lim().begin;
        X280Ctl xc{
            uint32_t(l2.x()),
            uint32_t(l2.y()),
            uint32_t(args_addr + 0x100),
            uint32_t(args_addr + 0x400)
        };
        auto env = [](const char* k, uint32_t d) {
            return std::getenv(k) ? uint32_t(std::atoi(std::getenv(k))) : d;
        };
        xc.group = env("TTX_GROUP", 0);
        xc.depth = env("TTX_DEPTH", 1);
        xc.rstride = env("TTX_RSTRIDE", 64);
        const uint64_t ordering = env("TTX_ORDER", 2);
        const uint32_t ncores = e.gx * e.gy;
        const uint32_t slots = xc.group ? e.gx : ncores;
        uint32_t trect[2][4] = {};
        for (size_t r = 0; r < e.panels.size(); r++) {
            const auto a = tr(e, e.panels[r].x0, 0),
                       b = tr(e, e.panels[r].x1, e.gy - 1);
            trect[r][0] = a.x;
            trect[r][1] = a.y;
            trect[r][2] = b.x;
            trect[r][3] = b.y;
        }
        for (int rep = 0; rep < 2; rep++) {  // rep 0 compiles
            xt.write_available_lim(xc.mailbox, std::vector<uint8_t>(64, 0));
            xt.write_available_lim(xc.acks, std::vector<uint8_t>(2048, 0));
            const uint64_t run = ++g_run;
            // L1 survives between processes: the check-in/gather tag must be unique per
            // process and run.
            xc.tag = ((std::random_device{}() ^ uint32_t(run)) & 0xFFFFu) | 1u;
            std::vector<uint8_t> block(192, 0);
            const uint64_t head[6] = {
                run, xc.mailbox, xc.acks, n, slots, e.panels.size()
            };
            std::memcpy(block.data(), head, sizeof(head));
            std::memcpy(block.data() + sizeof(head), trect, sizeof(trect));
            // ttx-sequencer.bin tail: scan=sweep, depth, ack stride, no host ring, host
            // depth, no host done, ordering, stride
            const uint64_t tail[8] = {1, xc.depth, 16, 0, 16, 0, ordering, xc.rstride};
            std::memcpy(
                block.data() + sizeof(head) + sizeof(trect), tail, sizeof(tail)
            );
            xt.write_available_lim(args_addr, block);
            ActiveRunRequest req;
            req.image_path = kImageDir + "/ttx-sequencer.bin";
            req.run_value = run;
            req.args_addr = args_addr;
            req.args_bytes = block.size();
            const auto enq = rt.enqueue(xt, req);
            if (!enq.session) {
                throw std::runtime_error("enqueue failed: " + describe(enq));
            }
            dist::MeshWorkload w;
            w.add_program(
                dist::MeshCoordinateRange(e.mesh->shape()),
                build(e, s, buf, n, false, xc)
            );
            const auto t0 = std::chrono::steady_clock::now();
            dist::EnqueueMeshWorkload(cq, w, false);
            const auto res = rt.session(enq.session)
                                 .wait_until_result_ready(std::chrono::seconds(60));
            const auto cleaned = rt.cleanup(enq.session, std::chrono::seconds(5));
            if (!res.succeeded() || !cleaned.cleaned_up()) {
                std::printf(
                    "   x280 FAILED: %s / %s (Finish may hang)\n",
                    describe(res).c_str(),
                    describe(cleaned).c_str()
                );
                std::fflush(stdout);
            }
            dist::Finish(cq);
            const double wall =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            if (!res.succeeded()) {
                return;
            }
            const uint64_t ticks = res.hartstates[0].state.result;
            if (ticks >= ~16ull) {
                std::printf(
                    "   x280 image error code %llu\n", (unsigned long long)~ticks
                );
                return;
            }
            if (rep == 1) {
                const double per = ticks / kMtimeHz / double(n);
                std::printf(
                    "   X280-sequenced [%s, ring %u]: %.2f us/matmul  %.1f TFLOPS  "
                    "(host wall %.1f ms for %u)\n",
                    xc.group ? "column write-gather" : "flat 120 acks",
                    xc.depth,
                    per * 1e6,
                    flop / per / 1e12,
                    wall * 1e3,
                    n
                );
                check("x280");
                x_per = per;
                if (const char* csvp = std::getenv("TTX_CSV")) {
                    if (FILE* f = std::fopen(csvp, "a")) {
                        std::fprintf(
                            f,
                            "matmul,%s_d%u,%u,%u,%u,%s,%.3f,%.2f\n",
                            xc.group ? "v1" : "flat",
                            xc.depth,
                            M,
                            N,
                            K,
                            std::getenv("TTX_PLL") ? std::getenv("TTX_PLL") : "1750",
                            per * 1e6,
                            flop / per / 1e12
                        );
                        std::fclose(f);
                    }
                }
            }
        }
    }
    if (variants.find('r') != std::string::npos ||
        variants.find('s') != std::string::npos) {
        run_program(e, build(e, s, buf, 1, true, none));  // compile + warm
        if (variants.find('r') != std::string::npos) {
            double best = 1e9;
            for (int i = 0; i < 3; i++) {
                best = std::min(best, run_program(e, build(e, s, buf, 1, true, none)));
            }
            std::printf(
                "   relaunch (slow dispatch, build+launch+finish per matmul): best "
                "%.1f ms\n",
                best * 1e3
            );
        }
        if (variants.find('s') != std::string::npos) {
            uint32_t ns = std::getenv("TTX_NS") ? std::atoi(std::getenv("TTX_NS")) : 0;
            if (ns == 0) {
                ns = x_per > 0 ? uint32_t(std::clamp(1.5 / x_per, 256.0, 200000.0))
                               : 20000;
            }
            run_program(e, build(e, s, buf, ns, true, none));  // compile
            double t1 = 1e9, tn = 1e9;
            for (int r = 0; r < 3; r++) {
                t1 = std::min(t1, run_program(e, build(e, s, buf, 1, true, none)));
                tn = std::min(tn, run_program(e, build(e, s, buf, ns, true, none)));
            }
            const double per = (tn - t1) / (ns - 1);
            std::printf(
                "   standalone loop (no sync, n=%u): %.2f us/matmul  %.1f TFLOPS\n",
                ns,
                per * 1e6,
                flop / per / 1e12
            );
            check("standalone");
        }
    }
    std::fflush(stdout);
}

}  // namespace

// Benchmarks the shapes given as arguments (or the defaults).
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<Shape> shapes;
    for (int i = 1; i < argc; i++) {
        Shape s{1, 1, 8, 2};
        std::sscanf(argv[i], "%u,%u,%u,%u", &s.pcM, &s.pcN, &s.Kt, &s.bw);
        shapes.push_back(s);
    }
    if (shapes.empty()) {
        shapes = {
            {1, 1, 8, 2}, {2, 2, 32, 4}, {4, 4, 64, 4}, {8, 8, 64, 4}, {8, 8, 128, 4}
        };
    }
    const uint32_t n = std::getenv("TTX_N") ? std::atoi(std::getenv("TTX_N")) : 64;
    const std::string variants =
        std::getenv("TTX_VARIANTS") ? std::getenv("TTX_VARIANTS") : "xsr";
    const std::string fid = std::getenv("TTX_FID") ? std::getenv("TTX_FID") : "HiFi2";
    const uint32_t pll =
        std::getenv("TTX_PLL") ? std::atoi(std::getenv("TTX_PLL")) : 1750;

    Env e;
    e.mesh = MeshDevice::create_unit_mesh(0);
    e.fid = fid == "LoFi"    ? mtl::MathFidelity::LoFi
            : fid == "HiFi3" ? mtl::MathFidelity::HiFi3
            : fid == "HiFi4" ? mtl::MathFidelity::HiFi4
                             : mtl::MathFidelity::HiFi2;
    e.fp32acc = std::getenv("TTX_FP32ACC") && std::atoi(std::getenv("TTX_FP32ACC"));
    const auto g = e.mesh->compute_with_storage_grid_size();
    e.gx = g.x;
    e.gy = g.y;
    // Panels: maximal runs of logical columns that are physically adjacent in NOC0.
    // Harvesting remaps translated columns, so a translated rectangle can physically
    // span harvested, L2CPU and DRAM columns.
    const auto chip = e.mesh->get_devices().front()->id();
    const auto& soc = mtl::MetalContext::instance().get_cluster().get_soc_desc(chip);
    auto phys_x = [&](uint32_t lx) {
        const auto t = tr(e, lx, 0);
        return soc
            .translate_coord_to(
                tt::umd::CoreCoord(
                    t.x, t.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED
                ),
                tt::CoordSystem::NOC0
            )
            .x;
    };
    for (uint32_t x = 0; x < e.gx; x++) {
        if (!e.panels.empty() && phys_x(x) == phys_x(e.panels.back().x1) + 1) {
            e.panels.back().x1 = x;
        } else {
            e.panels.push_back({x, x});
        }
    }
    std::printf("# grid %ux%u, panels:", e.gx, e.gy);
    for (const auto& pn : e.panels) {
        std::printf(" [x %u-%u]", pn.x0, pn.x1);
    }
    std::printf("  fidelity %s, bf16 in/out, n=%u\n", fid.c_str(), n);
    if (e.panels.size() > 2) {
        std::fprintf(stderr, "unexpected panel layout\n");
        return 1;
    }

    if (variants.find('x') == std::string::npos) {
        for (const auto& s : shapes) {
            bench_shape(e, nullptr, s, n, variants);
        }
        return 0;
    }
    auto opened = X280Runtime::open(
        e.mesh,
        RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared}
    );
    if (!opened.ready()) {
        std::fprintf(stderr, "X280 runtime not ready: %s\n", opened.diagnostic.c_str());
        return 2;
    }
    X280Runtime& rt = *opened.runtime;
    {
        bool all_idle = true;
        for (const auto& id : rt.tile_ids()) {
            all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
        }
        if (!all_idle) {
            MultiTileBootRequest req;
            req.tiles = rt.tile_ids();
            req.idle_image_path = kImageDir + "/firmware-lim-idle.bin";
            req.allow_reset_release = true;
            if (!rt.boot_tiles(req).ready()) {
                std::fprintf(stderr, "boot_tiles failed\n");
                return 2;
            }
        }
    }
    rt.set_pll(rt.tile_ids().front().chip_id, pll);
    for (const auto& s : shapes) {
        bench_shape(e, &rt, s, n, variants);
    }
    return 0;
}
