import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "独立测试_20261107_第五章强化学习改进" / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = {}
    manifest = {}
    for part in ("phaseC_part1", "phaseC_part2"):
        for path in sorted((SOURCE / part / "rows").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            label = row["job"]["label"]
            key = (label, row["scene"])
            if key in records:
                raise ValueError(f"Duplicate evaluation: {key}")
            records[key] = row
            manifest[str(path.relative_to(ROOT))] = digest(path)
    if len(records) != 384:
        raise ValueError(f"Expected 384 retained factorial rows, found {len(records)}")
    results = []
    for label in sorted({key[0] for key in records}):
        training = SOURCE / f"train_{label}"
        protocol = json.loads((training / "protocol.json").read_text(encoding="utf-8"))
        done = json.loads((training / "done.json").read_text(encoding="utf-8"))
        logs = [json.loads(line) for line in (training / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        selected_iteration = done["best"][1]
        selected = next(row for row in logs if row["iter"] == selected_iteration)
        if protocol["budget"] != 480000 or protocol["variant"] != "full":
            raise ValueError(f"Unmatched budget or observations: {label}")
        for name in ("protocol.json", "done.json", "best.pt", "train_log.jsonl"):
            manifest[str((training / name).relative_to(ROOT))] = digest(training / name)
        for split in ("E1n", "E2n"):
            rows = [row for (arm, scene), row in records.items() if arm == label and row["job"]["set"] == split]
            expected = 32 if split == "E1n" else 16
            if len(rows) != expected:
                raise ValueError(f"Incomplete split: {label}/{split}")
            if any(row["job"]["gate"] != "mu" or not row["job"]["envelope"] or row["sweeps"] != 8 for row in rows):
                raise ValueError(f"Unmatched inference configuration: {label}")
            times = [row["arrival"] for row in rows if row["status"] == "arrived" and row["audit_failed"] == 0]
            results.append(dict(
                label=label, split=split, seed=protocol["seed"], critic=protocol["critic"],
                projection_weight=protocol["proj"], allocated_steps=protocol["budget"],
                actual_training_steps=done["env_steps"], selected_iteration=selected_iteration,
                selected_steps=selected["env_steps"], count=len(rows), safe_arrivals=len(times),
                median_arrival=float(np.median(times)) if times else None,
                statuses=dict(Counter(row["status"] for row in rows)),
                audit_failed=sum(row["audit_failed"] for row in rows),
                cuts=sum(row["step_cuts"] for row in rows),
                restricted_completion_time=float(np.mean([row["arrival"] if row["status"] == "arrived" and row["audit_failed"] == 0 else 60.0 for row in rows])),
            ))
    pairs = []
    for critic in ("team", "local"):
        for seed_index in (1, 2):
            reference_label = f"c{critic}_p0_s{seed_index}"
            regularized_label = f"c{critic}_p1_s{seed_index}"
            reference = {scene: row for (label, scene), row in records.items() if label == reference_label}
            regularized = {scene: row for (label, scene), row in records.items() if label == regularized_label}
            if reference.keys() != regularized.keys():
                raise ValueError("Unpaired scenes")
            differences = []
            baseline_only = regularized_only = 0
            for scene in sorted(reference):
                left, right = reference[scene], regularized[scene]
                left_safe = left["status"] == "arrived" and left["audit_failed"] == 0
                right_safe = right["status"] == "arrived" and right["audit_failed"] == 0
                baseline_only += int(left_safe and not right_safe)
                regularized_only += int(right_safe and not left_safe)
                if left_safe and right_safe:
                    differences.append(right["arrival"] - left["arrival"])
            pairs.append(dict(critic=critic, seed_index=seed_index, baseline_only=baseline_only,
                              regularized_only=regularized_only, common=len(differences),
                              median_regularized_minus_baseline=float(np.median(differences))))
    output = HERE / "results_matched"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(dict(rows=results, pairs=pairs), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "source_sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 同预算、同种子的既有实验重统计", "", "回顾性分析；不是新增训练。每格保留两个训练种子，场景共享，不能视为独立算法重复。", "",
             "| 配置 | 集合 | 训练种子 | 实际训练步 | 选中步 | 安全到达 | 到达时间中位数/s | 失败按60s计的平均完成时间/s |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['label']} | {row['split']} | {row['seed']} | {row['actual_training_steps']} | {row['selected_steps']} | {row['safe_arrivals']}/{row['count']} | {row['median_arrival']:.2f} | {row['restricted_completion_time']:.2f} |")
    lines.extend(["", "所有384条记录均保留。训练预算按完整采样批次结束，实际步数略有超出；详细数值见表。未将仅两次训练的场景配对结果外推为算法级显著性。", "", "配对时间差为投影正则减基础策略，只在同一训练种子、同一价值网络和共同安全到达场景内计算：", ""])
    for row in pairs:
        lines.append(f"- {row['critic']} / s{row['seed_index']}：仅基础到达 {row['baseline_only']}，仅正则到达 {row['regularized_only']}，共同到达 {row['common']}，中位时间差 {row['median_regularized_minus_baseline']:+.2f} s。")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for language in ("zh", "en"):
        tex = []
        for critic in ("team", "local"):
            for weight in (0, 1):
                label = ("团队" if critic == "team" else "局部") if language == "zh" else ("Team" if critic == "team" else "Local")
                values = []
                for split in ("E1n", "E2n"):
                    selected_rows = [row for row in results if row["critic"] == critic and row["projection_weight"] == weight and row["split"] == split]
                    values.append(" / ".join(f"{row['safe_arrivals']}/{row['count']}" for row in selected_rows))
                pair_values = [row["median_regularized_minus_baseline"] for row in pairs if row["critic"] == critic]
                delta = " / ".join(f"{value:+.2f}" for value in pair_values) if weight else "---"
                tex.append(f"{label} & {weight} & {values[0]} & {values[1]} & {delta} " + r"\\")
        (output / f"matched_rows_{language}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps(dict(groups=len(results), evaluations=len(records), pairs=pairs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
