# SPDX-License-Identifier: Apache-2.0
"""Execute the production launch macro with immediate and deferred CPU handlers."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("workspace_bytes", [0, 4096])
def test_workspace_survives_command_submission(tmp_path, deferred, workspace_bytes):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required for the launch-macro lifetime regression")
    root = Path(__file__).resolve().parents[3]
    header = (root / "csrc/aclnn_torch_adapter/op_api_common.h").read_text()
    macro = header[header.index("#define EXEC_NPU_CMD(aclnn_api, ...)") :].rsplit("#endif", 1)[0]
    fixture = Path(__file__).with_name("op_api_workspace_lifetime.cpp").read_text()
    source = tmp_path / "lifetime.cpp"
    source.write_text(fixture.replace("// PRODUCTION_MACRO", macro))
    binary = tmp_path / "lifetime"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary), str(int(deferred)), str(workspace_bytes)], check=True, capture_output=True)
