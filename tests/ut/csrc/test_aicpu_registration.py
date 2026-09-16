# SPDX-License-Identifier: Apache-2.0
"""Build and package the production AICPU registration path without CANN."""

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


def _function(source, name):
    start = source.index(f"function({name}")
    end = source.index("endfunction()", start) + len("endfunction()")
    return source[start:end]


@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
@pytest.mark.parametrize("register_aicpu", [True, False], ids=["registered", "no_aicpu"])
def test_fresh_registration_packages_aicpu(tmp_path, generator, register_aicpu):
    # Exercise actual registration, gated generation, JSON merging and CPack;
    # only the device compiler, headers and runtime archives are host stand-ins.
    required = ("cmake", "cpack", "c++", "ar", "bash", "ninja" if generator == "Ninja" else "make")
    if any(not shutil.which(tool) for tool in required):
        pytest.skip("CMake/CPack, C++ compiler, ar, bash and selected build tool are required")
    csrc = Path(__file__).resolve().parents[3] / "csrc"
    source, build, ascend = (tmp_path / name for name in ("source", "build", "ascend"))
    source.mkdir()
    (source / "cmake").mkdir()
    (source / "scripts/util").mkdir(parents=True)
    for filename in ("merge_aicpu_info_json.sh", "insert_op_info.py", "const_var.py"):
        shutil.copyfile(csrc / "scripts/util" / filename, source / "scripts/util" / filename)
    symbol = (csrc / "cmake/symbol.cmake").read_text()
    (source / "cmake/symbol.cmake").write_text(
        _function(symbol, "gen_cust_aicpu_json_symbol")
        + "\n"
        + _function(symbol, "gen_cust_aicpu_kernel_symbol")
        + "\n"
    )
    registration = _function((csrc / "cmake/func.cmake").read_text(), "add_aicpu_cust_kernel_modules")
    top_level = (csrc / "CMakeLists.txt").read_text()
    start = top_level.index("        if (ENABLE_AICPU)")
    gate = top_level[start : top_level.index("        endif()", start) + len("        endif()")]
    compiler = ascend / "toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
    compiler.parent.mkdir(parents=True)
    compiler.symlink_to(shutil.which("c++"))
    archives = ascend / "lib64"
    archives.mkdir(parents=True)
    log = tmp_path / "commands.log"

    def run(*args):
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True)
        with log.open("a") as stream:
            stream.write(f"{list(map(str, args))!r}\n{result.stdout}\n{result.stderr}\n")
        result.check_returncode()
        return result

    for name in ("aicpu_context", "base_ascend_protobuf"):
        cpp, obj = (source / f"{name}.{suffix}" for suffix in ("cpp", "o"))
        cpp.write_text(f'extern "C" int {name}_value() {{ return 10; }}\n')
        run("c++", "-fPIC", "-c", cpp, "-o", obj)
        run("ar", "rcs", archives / f"lib{name}.a", obj)
    for name, value in (("first", 1), ("second", 2)):
        (source / f"{name}.cpp").write_text(
            'extern "C" int aicpu_context_value();\n'
            'extern "C" int base_ascend_protobuf_value();\n'
            f'extern "C" __attribute__((visibility("default"))) int {name}_value() {{\n'
            f"return {value} + aicpu_context_value() + base_ascend_protobuf_value(); }}\n"
        )
        (source / f"{name}.json").write_text(json.dumps({name: {"kernel": f"{name}_value"}}) + "\n")
    config = (
        "cmake_minimum_required(VERSION 3.20)\nproject(aicpu_registration CXX)\n"
        f'set(ASCEND_DIR "{ascend}")\n'
        "set(VENDOR_NAME test)\nset(OP_KERNEL_AICPU_UT ON)\n"
        "set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n"
        "add_library(intf_pub_cxx17 INTERFACE)\nadd_library(dlog_headers INTERFACE)\n" + registration + "\n"
    )
    if register_aicpu:
        for name in ("first", "second", "first"):
            config += f'add_aicpu_cust_kernel_modules({name} "${{CMAKE_SOURCE_DIR}}/{name}.cpp" '
            config += f'"${{CMAKE_SOURCE_DIR}}/{name}.json")\n'
    config += gate + '\ninstall(FILES first.cpp DESTINATION share)\nset(CPACK_GENERATOR "TGZ")\ninclude(CPack)\n'
    (source / "CMakeLists.txt").write_text(config)
    # The fresh configure deliberately does not pass ENABLE_AICPU.
    run("cmake", "-S", source, "-B", build, "-G", generator)
    prefix = "packages/vendors/test_transformer/op_impl/cpu/"
    expected = {prefix + "aicpu_kernel/impl/libtransformer_aicpu_kernels.so", prefix + "config/cust_aicpu_kernel.json"}

    def build_install_package(label):
        run("cmake", "--build", build, "-j", "2")
        install = tmp_path / f"install_{label}"
        run("cmake", "--install", build, "--prefix", install)
        installed = {str(p.relative_to(install)) for p in install.rglob("*") if p.is_file()}
        assert expected.intersection(installed) == (expected if register_aicpu else set())
        run("cmake", "--build", build, "--target", "package", "-j", "2")
        packages = list(build.glob("*.tar.gz"))
        assert len(packages) == 1
        with tarfile.open(packages[0]) as archive:
            names = archive.getnames()
        for name in expected:
            assert any(item.endswith("/" + name) for item in names) == register_aicpu
        if register_aicpu:
            cache = (build / "CMakeCache.txt").read_text()
            assert "ENABLE_AICPU:BOOL=ON" in cache
            assert json.loads((build / "cust_aicpu_kernel.json").read_text()) == {
                "first": {"kernel": "first_value"},
                "second": {"kernel": "second_value"},
            }
            library = install / prefix / "aicpu_kernel/impl/libtransformer_aicpu_kernels.so"
            result = run(
                sys.executable,
                "-c",
                "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); print(lib.first_value(),lib.second_value())",
                library,
            )
            assert result.stdout.strip() == "21 22"
        else:
            assert not (build / "libtransformer_aicpu_kernels.so").exists()
            assert not (build / "cust_aicpu_kernel.json").exists()

    build_install_package("fresh")
    watched = [build / name for name in ("libtransformer_aicpu_kernels.so", "cust_aicpu_kernel.json")]
    before = [p.stat().st_mtime_ns for p in watched] if register_aicpu else []
    # Reconfiguration must preserve registered artifacts without relying on a
    # previously enabled cache; registration overrides a stale explicit OFF.
    run("cmake", "-S", source, "-B", build, "-DENABLE_AICPU=OFF")
    build_install_package("reconfigured")
    assert ([p.stat().st_mtime_ns for p in watched] if register_aicpu else []) == before
