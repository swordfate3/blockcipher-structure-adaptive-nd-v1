from __future__ import annotations

import builtins
import importlib
import sys


PRELOADED_MATPLOTLIB_ARTIST = importlib.import_module("matplotlib.artist")


def test_export_checkpoint_scores_import_does_not_require_matplotlib(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, *args, **kwargs)

    for module_name in list(sys.modules):
        if module_name.startswith("blockcipher_nd.cli.export_checkpoint_scores"):
            monkeypatch.delitem(sys.modules, module_name)
        if module_name.startswith("blockcipher_nd.cli.evaluate_pairset_aggregation"):
            monkeypatch.delitem(sys.modules, module_name)
        if module_name.startswith("blockcipher_nd.evaluation"):
            monkeypatch.delitem(sys.modules, module_name)
        if module_name.startswith("matplotlib"):
            monkeypatch.delitem(sys.modules, module_name)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("blockcipher_nd.cli.export_checkpoint_scores")

    assert hasattr(module, "main")


def test_optional_import_probe_restores_preloaded_matplotlib_modules():
    assert sys.modules.get("matplotlib.artist") is PRELOADED_MATPLOTLIB_ARTIST
