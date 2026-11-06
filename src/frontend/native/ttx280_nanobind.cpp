// _ttx280: Python bindings for the X280 dispatch model v1, operating on ttnn's
// MeshDevice.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "ttx_dispatch_host.hpp"

namespace nb = nanobind;
using tt::tt_metal::distributed::MeshDevice;

namespace {

// Owns the X280 runtime for one ttnn MeshDevice.
struct Runtime {
    std::shared_ptr<MeshDevice> mesh;
    std::unique_ptr<x280::RuntimeOpenResult> opened;

    Runtime(std::shared_ptr<MeshDevice> mesh_device, uint32_t pll_mhz)
        : mesh(std::move(mesh_device)) {
        opened = std::make_unique<x280::RuntimeOpenResult>(x280::X280Runtime::open(
            mesh,
            x280::RuntimeOpenConfig{
                .priming_policy = x280::LimEccPrimingPolicy::RequirePrepared
            }
        ));
        if (!opened->ready()) {
            throw std::runtime_error("X280 runtime not ready: " + opened->diagnostic);
        }
        auto& rt = runtime();
        bool all_idle = true;
        for (const auto& id : rt.tile_ids()) {
            all_idle = all_idle && rt.tile(id).inspect().resident_idle_live;
        }
        if (!all_idle) {
            x280::MultiTileBootRequest req;
            req.tiles = rt.tile_ids();
            req.idle_image_path =
                std::string(X280_IMAGE_DIR) + "/firmware-lim-idle.bin";
            req.allow_reset_release = true;
            if (!rt.boot_tiles(req).ready()) {
                throw std::runtime_error("X280 boot_tiles failed");
            }
        }
        rt.set_pll(rt.tile_ids().front().chip_id, pll_mhz);
    }

    x280::X280Runtime& runtime() {
        return *opened->runtime;
    }
};

}  // namespace

NB_MODULE(_ttx280, m) {
    m.doc() =
        "X280 dispatch model v1 for Blackhole: drive resident Tensix kernels from the "
        "on-chip X280.";
    m.attr("CTL_CB") = ttx::kCtlCb;
    m.attr("TRACE_CB") = 10;
    m.attr("CTL_BYTES") = ttx::kCtlBytes;
    m.attr("MTIME_HZ") = ttx::kMtimeHz;
    m.attr("IMAGE_DIR") = std::string(X280_IMAGE_DIR);
    m.attr("KERNEL_DIR") = std::string(TTX_KERNEL_DIR);

    nb::class_<x280::SessionId>(m, "Session")
        .def_prop_ro("value", [](const x280::SessionId& s) {
            return s.value;
        });

    nb::class_<Runtime>(m, "Runtime")
        .def(
            nb::init<std::shared_ptr<MeshDevice>, uint32_t>(),
            nb::arg("mesh_device"),
            nb::arg("pll_mhz") = 1750,
            "Open the X280 runtime on a ttnn MeshDevice (slow dispatch, LIM ECC "
            "primed) and set PLL4."
        )
        .def(
            "plan",
            [](Runtime& self, uint32_t depth) {
                return ttx::DispatchPlan::make(self.mesh, self.runtime(), depth);
            },
            nb::arg("depth") = 1,
            nb::keep_alive<0, 1>(),
            "Build a dispatch plan: physically clean multicast panels, column groups, "
            "LIM layout."
        );

    nb::class_<ttx::DispatchPlan>(m, "Plan")
        .def_ro("grid_x", &ttx::DispatchPlan::gx)
        .def_ro("grid_y", &ttx::DispatchPlan::gy)
        .def_ro("panels", &ttx::DispatchPlan::panels)
        .def_ro("depth", &ttx::DispatchPlan::depth)
        .def_ro("tag", &ttx::DispatchPlan::tag)
        .def_rw(
            "trace_addr",
            &ttx::DispatchPlan::trace_addr,
            "DRAM trace tensor address; 0 = tracing off."
        )
        .def(
            "aiclk_mhz",
            &ttx::DispatchPlan::aiclk_mhz,
            "Chip AICLK in MHz (trace clock)."
        )
        .def(
            "translated",
            &ttx::DispatchPlan::translated,
            nb::arg("x"),
            nb::arg("y"),
            "Translated NOC0 coordinates of logical core (x, y)."
        )
        .def(
            "worker_args",
            &ttx::DispatchPlan::worker_args,
            nb::arg("x"),
            nb::arg("y"),
            "The 16 dispatch runtime args for the kernels of logical core (x, y)."
        )
        .def(
            "start",
            &ttx::DispatchPlan::start,
            nb::arg("n_cmds"),
            nb::arg("host_fed") = false,
            "Start the X280 sequencer for n_cmds commands; call before building worker "
            "args and launching."
        )
        .def(
            "finish",
            [](ttx::DispatchPlan& p, const x280::SessionId& session, double timeout_s) {
                nb::gil_scoped_release release;
                return p.finish(
                    session, std::chrono::seconds(static_cast<int64_t>(timeout_s))
                );
            },
            nb::arg("session"),
            nb::arg("timeout_s") = 60.0,
            "Wait for the sequencer; returns mtime ticks from command 1 to the last "
            "completion (0 = failure)."
        )
        .def(
            "submit",
            &ttx::DispatchPlan::submit,
            nb::arg("seq"),
            "Host-fed: enqueue command seq (1, 2, ...)."
        )
        .def(
            "wait",
            [](ttx::DispatchPlan& p, uint32_t seq) {
                nb::gil_scoped_release release;
                p.wait(seq);
            },
            nb::arg("seq"),
            "Host-fed: block until command seq completed on every core."
        )
        .def(
            "completed",
            &ttx::DispatchPlan::completed,
            "Host-fed: completed command count."
        );
}
