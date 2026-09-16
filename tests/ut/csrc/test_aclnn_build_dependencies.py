# SPDX-License-Identifier: Apache-2.0
"""Run the actual ACLNN generation rules against small compiled OpDef stand-ins."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("rule_file", ["CMakeLists.txt", "cmake/custom_build.cmake", "cmake/opbuild.cmake"])
@pytest.mark.parametrize("generator", ["Ninja", "Unix Makefiles"])
def test_opdef_change_regenerates_all_aclnn_variants(tmp_path, generator, rule_file):
    if any(not shutil.which(tool) for tool in ("cmake", "cc", "ninja" if generator == "Ninja" else "make")):
        pytest.skip("CMake, C compiler and the selected build tool are required")
    rule_path = Path(__file__).resolve().parents[3] / "csrc" / rule_file
    production = rule_path.read_text()
    if rule_file == "cmake/opbuild.cmake":
        rules = production[: production.index("function(append_versioned_aclnn_outputs")]
    else:
        start = production.index("    if (generate_aclnn_srcs)")
        end = production.index("    add_custom_target(opbuild_gen_exc", start)
        end = production.index("\n    )", end) + len("\n    )")
        rules = production[start:end]
    source, build = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    host = source / "host.c"
    host.write_text("int layout_policy(void) { return 1; }\n")
    tool = source / "opbuild"
    tool.write_text(
        "#!/usr/bin/env python3\nimport hashlib, os, sys\nfrom pathlib import Path\n"
        "host, out = map(Path, sys.argv[1:])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "variant = {'aclnn':'default', 'aclnnInner':'inner', 'aclnnExc':'exc'}[os.environ['OPS_PROJECT_NAME']]\n"
        "digest = hashlib.sha256(host.read_bytes()).hexdigest()\n"
        "(out / (variant + '.cpp')).write_text(digest)\n"
        "if variant == 'default': (out / 'default.h').write_text(digest)\n"
        "with (out / 'runs.txt').open('a') as log: log.write(digest + '\\n')\n"
    )
    tool.chmod(0o755)
    config = (
        "cmake_minimum_required(VERSION 3.20)\nproject(aclnn_regression C)\n"
        'set(base_aclnn_binary_dir "${CMAKE_BINARY_DIR}/autogen")\n'
        f'set(OP_BUILD_TOOL "{tool}")\n'
        'set(generate_aclnn_srcs "${base_aclnn_binary_dir}/default.cpp")\n'
        'set(generate_aclnn_headers "${base_aclnn_binary_dir}/default.h")\n'
        'set(generate_aclnn_inner_srcs "${base_aclnn_binary_dir}/inner/inner.cpp")\n'
        'set(generate_exclude_proto_srcs "${base_aclnn_binary_dir}/exc/exc.cpp")\n'
    )
    if rule_file == "cmake/opbuild.cmake":
        for dependency in ("intf_pub_cxx17", "exe_graph", "register", "c_sec"):
            config += f"add_library({dependency} INTERFACE)\n"
        config += rules
        for prefix, variant, subdir in (
            ("aclnn", "default", ""),
            ("aclnnInner", "inner", "inner"),
            ("aclnnExc", "exc", "exc"),
        ):
            headers = 'OUT_HEADERS "${base_aclnn_binary_dir}/default.h"' if variant == "default" else ""
            config += (
                f"gen_opbuild_target(TARGET opbuild_gen_{variant} PREFIX {prefix} GENACLNN 1 "
                f'IN_SRCS host.c OUT_DIR "${{base_aclnn_binary_dir}}" OUT_SUB_DIR "{subdir}" '
                f'OUT_SRCS "${{base_aclnn_binary_dir}}/{subdir}/{variant}.cpp" {headers})\n'
            )
    else:
        for variant in ("", "Inner", "Exc"):
            config += f"add_library(op_host_aclnn{variant} SHARED host.c)\n"
        config += rules
    config += "\nadd_custom_target(all_variants ALL)\n"
    config += "add_dependencies(all_variants opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)\n"
    (source / "CMakeLists.txt").write_text(config)

    def run(*args):
        subprocess.run(["cmake", *map(str, args)], check=True, capture_output=True, text=True)

    run("-S", source, "-B", build, "-G", generator)
    run("--build", build, "-j", "2")
    logs = [build / "autogen" / name for name in ("runs.txt", "inner/runs.txt", "exc/runs.txt")]
    before = [log.read_text().splitlines() for log in logs]
    assert all(len(lines) == 1 for lines in before)
    run("--build", build, "-j", "2")
    assert [log.read_text().splitlines() for log in logs] == before
    host.write_text("int layout_policy(void) { return 2; }\n")
    run("--build", build, "-j", "2")
    after = [log.read_text().splitlines() for log in logs]
    assert all(len(lines) == 2 and lines[0] != lines[1] for lines in after)
    run("--build", build, "-j", "2")
    assert [log.read_text().splitlines() for log in logs] == after
    # A changed generator must invalidate its generated files as well.
    tool.write_text(tool.read_text() + "# generator revision 2\n")
    run("--build", build, "-j", "2")
    assert all(len(log.read_text().splitlines()) == 3 for log in logs)
