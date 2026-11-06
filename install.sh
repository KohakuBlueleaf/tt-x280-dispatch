#!/usr/bin/env bash
# install.sh <tt-metal-v0.81.1-tree>: hooks this repository into a tt-metal source tree and
# builds the X280 images, the C++ hosts and the _ttx280 Python module from it.
# The tree must already be configured with -DTT_METAL_BUILD_X280=ON and ttnn (see README).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
tree=${1:?usage: install.sh <tt-metal tree>}
cmakelists=$tree/tt_metal/fabric/experimental/x280/CMakeLists.txt

# Add the include hook (three lines) once.
if ! grep -q TT_X280_DISPATCH_DIR "$cmakelists"; then
    git -C "$tree" apply "$here/patches/tt-metal-v0.81.1.patch"
fi

# Configure with this repository and build its targets.
cmake -B "$tree/build" -S "$tree" -DTT_X280_DISPATCH_DIR="$here" >/dev/null
ninja -C "$tree/build" x280_images ttx_bench ttx_matmul ttx_mlp _ttx280

echo "built; for Python add to the path next to ttnn:"
echo "  $tree/build/ttx280 and $here/src/frontend"
