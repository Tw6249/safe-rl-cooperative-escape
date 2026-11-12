import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "dgppo_scene_banks"
    output.mkdir(exist_ok=False)
    scope = runpy.run_path(str(HERE.parent / "experiments_sota_rl" / "run_sota_baselines.py"))
    environment = scope["E"]
    manifest = dict(training_agents=8, conditioning="Original train_scene conditioned on n=8",
                    novelty="New training bank; original development tasks are not a fresh final test",
                    banks={})
    for split, seed, count in (("train", 110101, 4096), ("validation", 110102, 48)):
        generator = np.random.default_rng(seed)
        scenes = []
        while len(scenes) < count:
            scene = environment.train_scene(generator)
            if scene["n"] == 8:
                scenes.append(scene)
        path = output / f"{split}.npz"
        np.savez_compressed(path, positions=np.array([scene["p"] for scene in scenes]),
                            goals=np.array([scene["goal"] for scene in scenes]),
                            thresholds=np.array([scene["lam"] for scene in scenes]))
        manifest["banks"][split] = dict(seed=seed, count=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    scenes = scope["all_96_scenes"]()
    for count in (8, 12, 20, 25):
        selected = [scene for scene in scenes.values() if scene["n"] == count]
        np.savez_compressed(output / f"development_n{count}.npz",
                            positions=np.array([scene["p"] for scene in selected]),
                            goals=np.array([scene["goal"] for scene in selected]),
                            thresholds=np.array([scene["lam"] for scene in selected]),
                            names=np.array([scene["name"] for scene in selected]))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
