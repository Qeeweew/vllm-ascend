# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE compressor_v41_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME CompressorV41
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE compressor_v41 ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
