import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "mixed_scene_banks"
    output.mkdir(exist_ok=False)
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    generator = np.random.default_rng(111001)
    banks = {count: [] for count in range(6, 11)}
    counts = {count: 0 for count in banks}
    while min(map(len, banks.values())) < 4096:
        scene = scope["E"].train_scene(generator)
        counts[scene["n"]] += 1
        if len(banks[scene["n"]]) < 4096:
            banks[scene["n"]].append(scene)
    manifest = dict(source="Frozen original train_scene, unchanged", seed=111001, accepted_counts=counts, files={})
    for count, scenes in banks.items():
        filename = output / f"train_n{count}.npz"
        np.savez_compressed(filename, positions=np.array([scene["p"] for scene in scenes]),
                            goals=np.array([scene["goal"] for scene in scenes]),
                            thresholds=np.array([scene["lam"] for scene in scenes]))
        manifest["files"][filename.name] = dict(count=len(scenes), sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
