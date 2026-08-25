add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE w4a16_moe_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME W4a16Moe
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE w4a16_moe ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
