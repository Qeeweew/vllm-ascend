# SPDX-License-Identifier: Apache-2.0
"""Select a single isolated OPP vendor for fused-QLI tests and benchmarks.

This runner only changes its own process. The production installation and
bootstrap implementation remain unchanged. The candidate package must include
the split gather/score operators when running the performance comparison.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opp-root", required=True, type=Path)
    parser.add_argument("mode", choices=("pytest", "script"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    opp = args.opp_root.resolve(strict=True)
    if not (opp / "op_api/lib/libcust_opapi.so").is_file():
        raise ValueError("Not an installed candidate vendor directory")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp)
    os.environ["LD_LIBRARY_PATH"] = str(opp / "op_api/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    from vllm_ascend import utils

    # Bootstrap normally prepends the production r12 package. CANN loads all
    # selected vendors' tiling registries, so this test process must select one.
    # Platform modules may already hold the original function by value. Its
    # global base directory must therefore be isolated too, not just the name.
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(opp)
    with (
        patch.object(utils, "_CUSTOM_OP_BASE_DIR", str(opp)),
        patch.object(utils, "bootstrap_custom_op_env", lambda **kwargs: None),
    ):
        if args.mode == "pytest":
            import pytest

            return pytest.main(args.arguments)
        if not args.arguments:
            parser.error("script mode requires a Python script path")
        sys.argv = args.arguments
        sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
        runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
