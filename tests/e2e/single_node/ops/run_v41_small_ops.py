# SPDX-License-Identifier: Apache-2.0
"""Load a complete isolated Torch extension with one optional candidate OPP.

Omit --opp-root only for CPU Meta/schema tests. Device tests must select the
vendor installed from their own complete operator-package build. Neither the
production extension nor the production OPP installation is modified.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--opp-root", type=Path)
    parser.add_argument(
        "--fallback-opp-root",
        type=Path,
        action="append",
        default=[],
        help="Additional installed vendors, searched after the candidate",
    )
    parser.add_argument("mode", choices=("pytest", "script"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extension = args.extension.resolve(strict=True)
    opp = args.opp_root.resolve(strict=True) if args.opp_root else None
    if args.fallback_opp_root and opp is None:
        parser.error("--fallback-opp-root requires --opp-root")
    vendors = ([opp] if opp else []) + [path.resolve(strict=True) for path in args.fallback_opp_root]
    if any(not (path / "op_api/lib/libcust_opapi.so").is_file() for path in vendors):
        raise ValueError("Every OPP root must be an installed vendor")
    vendor_path = ":".join(map(str, vendors))
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor_path
    if vendors:
        libraries = ":".join(str(path / "op_api/lib") for path in vendors)
        os.environ["LD_LIBRARY_PATH"] = libraries + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    import torch_npu  # noqa: F401 -- initialize Torch's PrivateUse1 registration

    from vllm_ascend import utils

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor_path
    # Also change the function's globals: platform.py may cache the original
    # bootstrap callable before it is patched by name.
    with (
        patch.object(utils, "_CUSTOM_OP_BASE_DIR", str(opp or extension.parent)),
        patch.object(utils, "bootstrap_custom_op_env", lambda **kwargs: None),
    ):
        module_name = "vllm_ascend.vllm_ascend_C"
        if module_name in sys.modules:
            raise RuntimeError("Torch extension was loaded before candidate isolation")
        spec = importlib.util.spec_from_file_location(module_name, extension)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the candidate Torch extension")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        sys.modules["vllm_ascend"].vllm_ascend_C = module
        print(
            json.dumps(
                {
                    "isolated_extension": str(extension),
                    "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
                    "candidate_opp": str(opp) if opp else None,
                    "ordered_opp_vendors": list(map(str, vendors)),
                }
            ),
            flush=True,
        )
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
