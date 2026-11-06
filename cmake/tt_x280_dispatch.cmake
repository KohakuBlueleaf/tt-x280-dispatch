# tt-x280-dispatch build, included from tt-metal's tt_metal/fabric/experimental/x280/CMakeLists.txt
# when TT_X280_DISPATCH_DIR is set (patches/tt-metal-v0.81.1.patch). Builds the X280 images,
# the C++ hosts and the _ttx280 Python module straight from this repository.

set(_ttx_src ${TT_X280_DISPATCH_DIR}/src)
set(TTX_KERNEL_DIR ${_ttx_src}/kernels CACHE INTERNAL "")
set(TTX_IMAGE_DIR ${X280_IMAGE_DIR} CACHE INTERNAL "")
set(TTX_PY_OUT ${PROJECT_BINARY_DIR}/ttx280 CACHE INTERNAL "")
set(TTX_HOST_INCLUDE ${_ttx_src}/dispatcher/host CACHE INTERNAL "")

# X280 images: the dispatch sequencer and the earlier experiment images.
x280_add_image(NAME ttx-sequencer SOURCES ${_ttx_src}/dispatcher/x280/sequencer.c)
x280_add_image(NAME ttx-compute SOURCES ${_ttx_src}/dispatcher/x280/experiments/compute.c)
x280_add_image(NAME ttx-pingpong SOURCES ${_ttx_src}/dispatcher/x280/experiments/pingpong.c)
x280_add_image(NAME ttx-sched SOURCES ${_ttx_src}/dispatcher/x280/experiments/sched.c)

# C++ hosts: protocol bench, matmul, MLP.
foreach(_host IN ITEMS bench:ttx_bench matmul:ttx_matmul mlp:ttx_mlp)
    string(REPLACE ":" ";" _pair ${_host})
    list(GET _pair 0 _file)
    list(GET _pair 1 _target)
    _x280_device_test(${_target} ${_ttx_src}/hosts/${_file}.cpp)
    target_include_directories(${_target} PRIVATE ${TTX_HOST_INCLUDE})
    target_compile_definitions(${_target} PRIVATE TTX_KERNEL_DIR="${TTX_KERNEL_DIR}")
endforeach()

# _ttx280 links ttnn's nanobind target (TT::Nanobind), defined after tt_metal: add it last.
function(_ttx_add_python_module)
    if(NOT TARGET TT::Nanobind)
        message(STATUS "tt-x280-dispatch: ttnn nanobind not configured, skipping _ttx280")
        return()
    endif()
    add_library(_ttx280 MODULE ${TT_X280_DISPATCH_DIR}/src/frontend/native/ttx280_nanobind.cpp)
    target_include_directories(
        _ttx280
        PRIVATE
            ${TTX_HOST_INCLUDE}
            ${X280_SOURCE_DIR}
            ${X280_SOURCE_DIR}/src/device
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
    )
    target_link_libraries(_ttx280 PRIVATE TT::Nanobind x280_device TT::Metalium)
    target_compile_definitions(
        _ttx280
        PRIVATE
            X280_IMAGE_DIR="${TTX_IMAGE_DIR}"
            TTX_KERNEL_DIR="${TTX_KERNEL_DIR}"
    )
    nanobind_opt_size(_ttx280)
    nanobind_compile_options(_ttx280)
    nanobind_link_options(_ttx280)
    nanobind_extension_abi3(_ttx280)
    set_target_properties(_ttx280 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TTX_PY_OUT})
    add_dependencies(_ttx280 x280_images)
endfunction()

set(X280_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} CALL _ttx_add_python_module)
