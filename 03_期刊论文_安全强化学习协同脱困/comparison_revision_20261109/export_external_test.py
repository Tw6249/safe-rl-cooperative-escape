import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "external_test"
    output.mkdir(exist_ok=False)
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    environment = scope["E"]
    scenes = environment.lattice_scenes(111002, n=8, n_local=20, n_perm=12, fracs=(0.6, 0.9))[0]
    for scene in scenes.values():
        scene["bench"] = "E1new"
    for seed in (111003, 111004):
        more = environment.scenes_for(seed)
        for scene in more.values():
            scene["bench"] = "E2new"
            scene.setdefault("frac", 0.6)
        scenes.update(more)
    if len(scenes) != 96:
        raise ValueError("Expected 96 new external-comparison tasks")
    serializable = []
    for scene in sorted(scenes.values(), key=lambda row: row["name"]):
        serializable.append({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in scene.items()})
    text = json.dumps(serializable, ensure_ascii=False, indent=2)
    (output / "scenes.json").write_text(text, encoding="utf-8")
    for count in (8, 12, 20, 25):
        selected = [scene for scene in scenes.values() if scene["n"] == count]
        np.savez_compressed(output / f"n{count}.npz", positions=np.array([scene["p"] for scene in selected]),
                            goals=np.array([scene["goal"] for scene in selected]),
                            thresholds=np.array([scene["lam"] for scene in selected]),
                            names=np.array([scene["name"] for scene in selected]))
    (output / "manifest.json").write_text(json.dumps(dict(count=96, task_seed_streams=[111002,111003,111004],
        selection="Final-budget model for both methods; this test does not select checkpoints or hyperparameters",
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
