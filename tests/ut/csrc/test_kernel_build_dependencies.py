# SPDX-License-Identifier: Apache-2.0
"""Exercise production CMake copy/compile stamps without a CANN installation."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
def test_nested_kernel_edits_recopy_and_recompile(tmp_path, generator):
    if not shutil.which("cmake") or not shutil.which("ninja" if generator == "Ninja" else "make"):
        pytest.skip("CMake and the selected build tool are required")
    production = Path(__file__).resolve().parents[3] / "csrc/cmake/func.cmake"
    source = tmp_path / "source"
    nested = source / "foo/op_kernel/arch22/nested.h"
    nested.parent.mkdir(parents=True)
    nested.write_text("kernel-v1\n")
    dynamic = source / "impl/dynamic/foo.py"
    dynamic.parent.mkdir(parents=True)
    dynamic.write_text("metadata-v1\n")
    build = tmp_path / "build"
    script = build / "binary/ascend910b/gen/Foo-foo-0.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        '#!/bin/bash\nset -eu\ncat "${1%/*}/op_kernel/arch22/nested.h" "$1" > "$2/result.txt"\n'
        'echo compiled >> "$2/runs.txt"\n'
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(stamp_regression NONE)\n"
        f'include("{production}")\n'
        'set(OPS_ADV_UTILS_KERNEL_INC "${CMAKE_SOURCE_DIR}/absent")\n'
        'set(ASCEND_BINARY_OUT_DIR "${CMAKE_BINARY_DIR}/binary")\n'
        'set(ASCEND_IMPL_OUT_DIR "${CMAKE_SOURCE_DIR}/impl")\n'
        'set(VENDOR_NAME "test")\n'
        "add_custom_target(ops_transformer_kernel)\nadd_custom_target(ops_transformer_config)\n"
        'add_bin_compile_target(COMPUTE_UNIT ascend910b OP_INFO "${CMAKE_SOURCE_DIR}/foo")\n'
    )

    def run(*args):
        subprocess.run(["cmake", *map(str, args)], check=True, capture_output=True, text=True)

    def compile_kernel():
        run("--build", build, "--target", "foo_ascend910b_0", "-j", "2")

    run("-S", source, "-B", build, "-G", generator)
    output = build / "binary/ascend910b/bin/foo"
    compile_kernel()
    assert (output / "result.txt").read_text() == "kernel-v1\nmetadata-v1\n"
    compile_kernel()
    assert (output / "runs.txt").read_text().splitlines() == ["compiled"]

    nested.write_text("kernel-v2\n")
    compile_kernel()
    assert (output / "result.txt").read_text() == "kernel-v2\nmetadata-v1\n"
    dynamic.write_text("metadata-v2\n")
    compile_kernel()
    assert (output / "result.txt").read_text() == "kernel-v2\nmetadata-v2\n"

    added = nested.with_name("added.h")
    added.write_text("new nested header\n")
    compile_kernel()
    copied = build / "binary/ascend910b/src/foo/op_kernel/arch22/added.h"
    assert copied.read_bytes() == added.read_bytes()
    assert len((output / "runs.txt").read_text().splitlines()) == 4
