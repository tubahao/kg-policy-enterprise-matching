# -*- coding: utf-8 -*-
"""
每次 pip install dgl 后运行（无需先能 import dgl）：
- 去掉 dataloader 对 DistGraph 的顶层 import
- dist_dataloader / distributed 导入失败时跳过（GraphBolt 与 PyTorch 不匹配）

用法（在项目根目录下）:
  python scripts/apply_dgl_windows_single_gpu_patch.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def site_dgl() -> Path:
    p = Path(sys.executable).resolve()
    roots = [p.parent, p.parent.parent]
    for root in roots:
        cand = root / "Lib" / "site-packages" / "dgl"
        if cand.is_dir():
            return cand
    raise SystemExit(f"未找到 dgl（已试环境根: {roots}）")


SITE = site_dgl()


def patch_dataloader() -> None:
    p = SITE / "dataloading" / "dataloader.py"
    c = p.read_text(encoding="utf-8")
    old_imp = (
        "from ..cuda import GPUCache\n"
        "from ..distributed import DistGraph\n"
        "from ..frame import LazyFeature\n"
    )
    new_imp = (
        "from ..cuda import GPUCache\n"
        "from ..frame import LazyFeature\n"
    )
    if old_imp in c:
        c = c.replace(old_imp, new_imp, 1)
    old_is = """        if isinstance(graph, DistGraph):
            raise TypeError(
                "Please use dgl.dataloading.DistNodeDataLoader or "
                "dgl.datalaoding.DistEdgeDataLoader for DistGraphs."
            )
"""
    new_is = """        _mod = getattr(graph.__class__, "__module__", "") or ""
        if graph.__class__.__name__ == "DistGraph" and _mod.startswith("dgl.distributed"):
            raise TypeError(
                "Please use dgl.dataloading.DistNodeDataLoader or "
                "dgl.datalaoding.DistEdgeDataLoader for DistGraphs."
            )
"""
    if old_is in c:
        c = c.replace(old_is, new_is, 1)
    p.write_text(c, encoding="utf-8")
    print("ok:", p)


def patch_dataloading_init() -> None:
    p = SITE / "dataloading" / "__init__.py"
    c = p.read_text(encoding="utf-8")
    plain = """if F.get_preferred_backend() == "pytorch":
    from .spot_target import *
    from .dataloader import *
    from .dist_dataloader import *
"""
    wrapped = """if F.get_preferred_backend() == "pytorch":
    from .spot_target import *
    from .dataloader import *
    try:
        from .dist_dataloader import *
    except (ImportError, FileNotFoundError, OSError):
        pass
"""
    nl = c.replace("\r\n", "\n")
    if wrapped.strip() in nl:
        print("skip:", p.name)
        return
    if plain not in c:
        raise SystemExit(f"{p}: 未找到预期未补丁块")
    c = c.replace(plain, wrapped, 1)
    p.write_text(c, encoding="utf-8")
    print("ok:", p)


def patch_dgl_init() -> None:
    p = SITE / "__init__.py"
    c = p.read_text(encoding="utf-8")
    plain = """if backend_name == "pytorch":
    from . import distributed
"""
    wrapped = """if backend_name == "pytorch":
    try:
        from . import distributed
    except (ImportError, FileNotFoundError, OSError):
        import warnings

        warnings.warn(
            "dgl.distributed 未加载（GraphBolt 问题时可忽略）。单机 GNN 仍可用。",
            ImportWarning,
            stacklevel=2,
        )
"""
    if "try:\n        from . import distributed" in c.replace("\r\n", "\n"):
        print("skip:", p.name)
        return
    if plain not in c:
        raise SystemExit(f"{p}: 未找到预期未补丁块")
    c = c.replace(plain, wrapped, 1)
    p.write_text(c, encoding="utf-8")
    print("ok:", p)


def main() -> None:
    patch_dataloader()
    patch_dataloading_init()
    patch_dgl_init()
    print("完成。SITE =", SITE)


if __name__ == "__main__":
    main()
