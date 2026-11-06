/* X280 compute microbenchmark. Args (LIM cache line): run_value, mode, iters. Modes: 0
 * fp32 vfmacc m1 x8, 1 fp32 vfmacc m8 x2, 2 int32 vmacc m1 x8, 3 scalar fmadd.s x8.
 * Result per hart: mtime ticks (50 MHz); ~0 = stale args. */
#include <stdint.h>

#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_hart_protocol.h"

typedef struct {
    uint64_t run_value;
    uint64_t mode;
    uint64_t iters;
} ttx_args_t;

/* n rounds of 8 fp32 vfmacc at LMUL 1. */
static void run_fp32_m1(uint64_t n) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetvli t0, zero, e32, m1, ta, ma\n"
        "li t1, 1\nfcvt.s.w ft0, t1\n"
        "vfmv.v.f v8, ft0\nvfmv.v.f v9, ft0\n"
        "vmv.v.i v0, 0\nvmv.v.i v1, 0\nvmv.v.i v2, 0\nvmv.v.i v3, 0\n"
        "vmv.v.i v4, 0\nvmv.v.i v5, 0\nvmv.v.i v6, 0\nvmv.v.i v7, 0\n"
        "1:\n"
        "vfmacc.vv v0, v8, v9\nvfmacc.vv v1, v8, v9\nvfmacc.vv v2, v8, v9\nvfmacc.vv "
        "v3, v8, v9\n"
        "vfmacc.vv v4, v8, v9\nvfmacc.vv v5, v8, v9\nvfmacc.vv v6, v8, v9\nvfmacc.vv "
        "v7, v8, v9\n"
        "addi %0, %0, -1\nbnez %0, 1b\n"
        ".option pop\n"
        : "+r"(n)
        :
        : "t0", "t1", "ft0", "memory"
    );
}

/* n rounds of 2 fp32 vfmacc at LMUL 8. */
static void run_fp32_m8(uint64_t n) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetvli t0, zero, e32, m8, ta, ma\n"
        "li t1, 1\nfcvt.s.w ft0, t1\n"
        "vfmv.v.f v16, ft0\nvfmv.v.f v24, ft0\n"
        "vmv.v.i v0, 0\nvmv.v.i v8, 0\n"
        "1:\n"
        "vfmacc.vv v0, v16, v24\nvfmacc.vv v8, v16, v24\n"
        "addi %0, %0, -1\nbnez %0, 1b\n"
        ".option pop\n"
        : "+r"(n)
        :
        : "t0", "t1", "ft0", "memory"
    );
}

/* n rounds of 8 int32 vmacc at LMUL 1. */
static void run_i32_m1(uint64_t n) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetvli t0, zero, e32, m1, ta, ma\n"
        "vmv.v.i v8, 1\nvmv.v.i v9, 1\n"
        "vmv.v.i v0, 0\nvmv.v.i v1, 0\nvmv.v.i v2, 0\nvmv.v.i v3, 0\n"
        "vmv.v.i v4, 0\nvmv.v.i v5, 0\nvmv.v.i v6, 0\nvmv.v.i v7, 0\n"
        "1:\n"
        "vmacc.vv v0, v8, v9\nvmacc.vv v1, v8, v9\nvmacc.vv v2, v8, v9\nvmacc.vv v3, "
        "v8, v9\n"
        "vmacc.vv v4, v8, v9\nvmacc.vv v5, v8, v9\nvmacc.vv v6, v8, v9\nvmacc.vv v7, "
        "v8, v9\n"
        "addi %0, %0, -1\nbnez %0, 1b\n"
        ".option pop\n"
        : "+r"(n)
        :
        : "t0", "memory"
    );
}

/* n rounds of 8 scalar fmadd.s. */
static void run_scalar_fp32(uint64_t n) {
    __asm__ volatile(
        "li t1, 1\nfcvt.s.w ft8, t1\nfmv.s ft9, ft8\n"
        "fmv.w.x ft0, zero\nfmv.w.x ft1, zero\nfmv.w.x ft2, zero\nfmv.w.x ft3, zero\n"
        "fmv.w.x ft4, zero\nfmv.w.x ft5, zero\nfmv.w.x ft6, zero\nfmv.w.x ft7, zero\n"
        "1:\n"
        "fmadd.s ft0, ft8, ft9, ft0\nfmadd.s ft1, ft8, ft9, ft1\nfmadd.s ft2, ft8, "
        "ft9, ft2\nfmadd.s ft3, ft8, ft9, "
        "ft3\n"
        "fmadd.s ft4, ft8, ft9, ft4\nfmadd.s ft5, ft8, ft9, ft5\nfmadd.s ft6, ft8, "
        "ft9, ft6\nfmadd.s ft7, ft8, ft9, "
        "ft7\n"
        "addi %0, %0, -1\nbnez %0, 1b\n"
        : "+r"(n)
        :
        : "t1",
          "ft0",
          "ft1",
          "ft2",
          "ft3",
          "ft4",
          "ft5",
          "ft6",
          "ft7",
          "ft8",
          "ft9",
          "memory"
    );
}

/* mstatus.FS (bits 13-14) and mstatus.VS (bits 9-10) set to Dirty: enables the FPU and
 * vector unit. */
static void enable_fp_vector(void) {
    const uint64_t bits = (3ull << 13) | (3ull << 9);
    __asm__ volatile("csrs mstatus, %0" : : "r"(bits) : "memory");
}

/* Runs the selected mode on every hart and stores its mtime ticks. */
int main(uint64_t hartid) {
    enable_fp_vector();
    const volatile ttx_args_t* args =
        (const volatile ttx_args_t*)x280_hart_args(hartid);
    if (args == 0 || args->run_value != x280_hart_control(hartid)->run_value) {
        x280_hart_set_result(hartid, ~0ull);
        return 0;
    }
    const uint64_t mode = args->mode;
    const uint64_t iters = args->iters;
    const uint64_t t0 = x280_mtime();
    switch (mode) {
        case 0:
            run_fp32_m1(iters);
            break;
        case 1:
            run_fp32_m8(iters);
            break;
        case 2:
            run_i32_m1(iters);
            break;
        case 3:
            run_scalar_fp32(iters);
            break;
        default:
            x280_hart_set_result(hartid, ~1ull);
            return 0;
    }
    x280_hart_set_result(hartid, x280_mtime() - t0);
    return 0;
}
