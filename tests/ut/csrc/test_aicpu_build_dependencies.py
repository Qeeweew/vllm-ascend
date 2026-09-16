# SPDX-License-Identifier: Apache-2.0
"""Exercise the production AICPU link rule with native stand-in objects."""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
@pytest.mark.parametrize("library_dir", ["ops_base/lib64", "lib64"])
def test_aicpu_object_and_archive_changes_relink(tmp_path, generator, library_dir):
    required = ("cmake", "c++", "ar", "ninja" if generator == "Ninja" else "make")
    if any(not shutil.which(tool) for tool in required):
        pytest.skip("CMake, a C++ compiler, ar and the selected build tool are required")
    production = (Path(__file__).resolve().parents[3] / "csrc/cmake/symbol.cmake").read_text()
    start = production.index("function(gen_cust_aicpu_kernel_symbol)")
    end = production.index("endfunction()", start) + len("endfunction()")
    source, build, ascend = (tmp_path / name for name in ("source", "build", "ascend"))
    source.mkdir()
    compiler = ascend / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
    compiler.parent.mkdir(parents=True)
    compiler.symlink_to(shutil.which("c++"))
    archives = ascend / library_dir
    archives.mkdir(parents=True)

    def run(*args):
        return subprocess.run(list(map(str, args)), check=True, capture_output=True, text=True)

    def write_archive(name, value):
        cpp, obj = (source / f"{name}.{suffix}" for suffix in ("cpp", "o"))
        cpp.write_text(f'extern "C" int {name}_value() {{ return {value}; }}\n')
        run("c++", "-fPIC", "-c", cpp, "-o", obj)
        run("ar", "rcs", archives / f"lib{name}.a", obj)

    def write_kernel(value):
        (source / "kernel.cpp").write_text(
            'extern "C" int aicpu_context_value();\n'
            'extern "C" int base_ascend_protobuf_value();\n'
            'extern "C" int kernel_value() {\n'
            f"  return {value} + aicpu_context_value() + base_ascend_protobuf_value();\n"
            "}\n"
        )

    write_archive("aicpu_context", 10)
    write_archive("base_ascend_protobuf", 100)
    write_kernel(1)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(aicpu_link_regression CXX)\n"
        f'set(ASCEND_DIR "{ascend}")\n'
        "set(VENDOR_NAME test)\n"
        "add_library(kernel OBJECT kernel.cpp)\n"
        "set_target_properties(kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)\n"
        "set(AICPU_CUST_OBJ_TARGETS kernel)\n" + production[start:end] + "\ngen_cust_aicpu_kernel_symbol()\n"
    )
    run("cmake", "-S", source, "-B", build, "-G", generator)
    output = build / "libtransformer_aicpu_kernels.so"

    def build_and_observe(expected):
        run("cmake", "--build", build, "-j", "2")
        # A fresh process avoids dlopen caching a previously loaded library.
        result = run(
            sys.executable,
            "-c",
            "import ctypes, sys; print(ctypes.CDLL(sys.argv[1]).kernel_value())",
            output,
        )
        assert int(result.stdout) == expected
        return output.stat().st_mtime_ns, hashlib.sha256(output.read_bytes()).hexdigest()

    initial = build_and_observe(111)
    assert build_and_observe(111) == initial, "An unchanged build must not relink"
    write_kernel(2)
    changed_kernel = build_and_observe(112)
    assert changed_kernel[0] > initial[0] and changed_kernel[1] != initial[1]
    assert build_and_observe(112) == changed_kernel
    write_archive("aicpu_context", 20)
    changed_context = build_and_observe(122)
    assert changed_context[0] > changed_kernel[0] and changed_context[1] != changed_kernel[1]
    assert build_and_observe(122) == changed_context
    write_archive("base_ascend_protobuf", 200)
    changed_protobuf = build_and_observe(222)
    assert changed_protobuf[0] > changed_context[0] and changed_protobuf[1] != changed_context[1]
    assert build_and_observe(222) == changed_protobuf
