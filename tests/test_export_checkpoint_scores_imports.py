from __future__ import annotations

import builtins
import importlib
import sys


def test_export_checkpoint_scores_import_does_not_require_matplotlib(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, *args, **kwargs)

    for module_name in list(sys.modules):
        if module_name.startswith("blockcipher_nd.cli.export_checkpoint_scores"):
            sys.modules.pop(module_name)
        if module_name.startswith("blockcipher_nd.cli.evaluate_pairset_aggregation"):
            sys.modules.pop(module_name)
        if module_name.startswith("blockcipher_nd.evaluation"):
            sys.modules.pop(module_name)
        if module_name.startswith("matplotlib"):
            sys.modules.pop(module_name)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("blockcipher_nd.cli.export_checkpoint_scores")

    assert hasattr(module, "main")
