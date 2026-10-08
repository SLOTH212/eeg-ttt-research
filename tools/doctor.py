
"""Read-only environment inventory; never imports experiment code or downloads assets."""
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    packages={}
    for name in ["torch","numpy","scipy","pandas","scikit-learn","matplotlib","pyedflib","requests","einops","timm"]:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    paths=["data/processed","phase2/processed","phase6/processed","phase62/upstream",
        "phase62/pretrained_weights.pth","phase79/protocol.json","phase85/protocol.json",
        "phase119/history11_seed17/model.pt","phase121/history11_seed17/model.pt","phase135/history11_seed17/model.pt"]
    allowed=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None
    print(json.dumps(dict(python=sys.version.split()[0],platform=sys.platform,
        current_cpu_affinity=allowed,historical_cpu_ids=list(range(8,24)),
        requested_cpu_ids_available=allowed is not None and set(range(8,24))<=set(allowed),
        package_versions=packages,local_assets={p:(ROOT/p).exists() for p in paths},
        missing_assets_expected_in_code_only_release=True,downloads=False,
        experiment_modules_imported=False,research_status="PAUSED_BY_USER"),indent=2))
if __name__ == "__main__": main()
