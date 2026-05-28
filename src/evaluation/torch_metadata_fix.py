# -*- coding: utf-8 -*-
"""
部分 Conda / 混合 pip 安装的 PyTorch 会导致 importlib.metadata 中 torch 分发元数据损坏：
`importlib.metadata.version("torch")` 返回 None，而 `torch.__version__` 仍正常。
transformers 在导入时用 version.parse(torch_version)，对 None 会 TypeError。

在任意 `import transformers` 之前调用 apply_torch_metadata_fix() 即可。
根治：conda 环境内 `pip install --force-reinstall torch` 或重装 pytorch 包。
"""

from __future__ import annotations

import importlib.metadata as im
from typing import Callable

_applied: bool = False
_orig_version: Callable[[str], str] | None = None


def apply_torch_metadata_fix() -> None:
    global _applied, _orig_version
    if _applied:
        return

    need_patch = False
    try:
        v = im.version("torch")
        if v is None or (isinstance(v, str) and not str(v).strip()):
            need_patch = True
    except im.PackageNotFoundError:
        need_patch = True

    if not need_patch:
        _applied = True
        return

    try:
        import torch as _torch
    except ImportError:
        _applied = True
        return

    tv = getattr(_torch, "__version__", None)
    fallback = str(tv).strip() if tv is not None and str(tv).strip() else "2.0.0"

    if _orig_version is None:
        _orig_version = im.version

    def _version(name: str) -> str:
        assert _orig_version is not None
        if name == "torch":
            try:
                r = _orig_version(name)
            except im.PackageNotFoundError:
                r = None
            if r is None or (isinstance(r, str) and not str(r).strip()):
                return fallback
            return str(r)
        return _orig_version(name)

    im.version = _version  # type: ignore[method-assign]
    _applied = True
