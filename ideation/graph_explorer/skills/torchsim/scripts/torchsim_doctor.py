#!/usr/bin/env python3
from __future__ import annotations
import importlib, json, platform, sys

def try_import(name: str):
    try:
        mod = importlib.import_module(name); return {"ok": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc: return {"ok": False, "error": repr(exc)}

def main() -> None:
    report = {"python": sys.version, "python_version_tuple": list(sys.version_info[:3]), "platform": platform.platform(), "imports": {}}
    for name in ["torch", "torch_sim", "ase", "pymatgen", "phonopy", "mace", "mattersim", "orb_models", "sevenn", "fairchem", "nequip", "nequix", "h5py", "tables", "numpy", "matplotlib"]:
        report["imports"][name] = try_import(name)
    try:
        import torch
        report["torch_cuda"] = {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [], "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
    except Exception as exc: report["torch_cuda"] = {"error": repr(exc)}
    try:
        import torch_sim as ts
        report["torchsim_api"] = {"has_integrate": hasattr(ts, "integrate"), "has_optimize": hasattr(ts, "optimize"), "has_static": hasattr(ts, "static"), "integrators": [x for x in dir(ts.Integrator) if not x.startswith("_")] if hasattr(ts, "Integrator") else [], "optimizers": [x for x in dir(ts.Optimizer) if not x.startswith("_")] if hasattr(ts, "Optimizer") else []}
    except Exception as exc: report["torchsim_api"] = {"error": repr(exc)}
    print(json.dumps(report, indent=2))
    if sys.version_info < (3, 12): raise SystemExit("TorchSim requires Python >= 3.12")
if __name__ == "__main__": main()
