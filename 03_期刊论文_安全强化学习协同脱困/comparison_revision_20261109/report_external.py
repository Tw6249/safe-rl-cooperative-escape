import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    groups = []
    for method in ("egp", "dgppo"):
        for seed_index in (1, 2):
            directory = HERE / f"external_{method}_s{seed_index}"
            candidates = [HERE / f"external_audit_v2_{method}_s{seed_index}", HERE / f"external_audit_{method}_s{seed_index}", directory]
            audited = next(folder for folder in candidates if (folder / "summary_audited.json").exists())
            rows = json.loads((audited / "summary_audited.json").read_text(encoding="utf-8"))
            if len(rows) != 96 or len({row["scene"] for row in rows}) != 96:
                raise ValueError(f"Incomplete or repeated test: {directory}")
            for row in rows:
                if "final_max_err" not in row:
                    suffix = "_egp_final" if method == "egp" else ""
                    with np.load(directory / f"{row['scene']}{suffix}.npz") as archive:
                        row["final_max_err"] = float(np.linalg.norm(archive["states"][-1] - archive["goal"], axis=1).max())
            for split, selected in (("N8", [row for row in rows if row["n"] == 8]),
                                    ("Cross", [row for row in rows if row["n"] != 8])):
                times = [row["arrival"] for row in selected if row["safe_arrived"]]
                groups.append(dict(method=method, seed_index=seed_index, split=split,
                    count=len(selected), safe_arrivals=sum(row["safe_arrived"] for row in selected),
                    arrived=sum(row["status"] == "arrived" for row in selected),
                    failed_audit_runs=sum(row["audit_failed"] > 0 for row in selected),
                    collision_runs=sum(row.get("numeric_min_distance", row["audit_min_distance"]) < 1 - 1e-9 for row in selected),
                    statuses=dict(Counter(row["status"] for row in selected)),
                    median_safe_arrival=float(np.median(times)) if times else None,
                    failure_penalized_completion=float(np.mean([row["arrival"] if row["safe_arrived"] else 60.0 for row in selected])),
                    minimum_distance=min(row["audit_min_distance"] for row in selected),
                    final_error_median=float(np.median([row["final_max_err"] for row in selected])),
                    audit_failed_segments=sum(row["audit_failed"] for row in selected)))
    text = ["# 独立DGPPO适配与EGP-NPA的受限端到端比较", "",
            "每方法两个训练种子、约48万环境交互，固定最终预算检查点；使用相同新96任务与独立审计。训练目标、有限场景库、观测和原生网络存在披露差异，不作组件因果结论或算法级显著性结论。", "",
            "| 方法 | 训练重复 | 测试 | 安全到达 | 名义到达 | 审计失败运行 | 失败按60s计的平均时间/s | 最小距离/m |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in groups:
        text.append(f"| {row['method']} | {row['seed_index']} | {row['split']} | {row['safe_arrivals']}/{row['count']} | {row['arrived']} | {row['failed_audit_runs']} | {row['failure_penalized_completion']:.2f} | {row['minimum_distance']:.4f} |")
    (HERE / "external_summary.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    (HERE / "external_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    for language in ("zh", "en"):
        header = "方法 & 重复 & $N=8$ 安全到达 & 跨规模安全到达 & 审计失败运行 & 平均完成时间/s" if language == "zh" else "Method & Replicate & $N=8$ Safe & Cross-Scale Safe & Failed-Audit Runs & Mean Completion/s"
        caption = "新生成96任务上的外部方法适配比较。平均完成时间将未安全到达计为60秒；审计失败不自动等同于物理碰撞。" if language == "zh" else "Adapted External-Method Comparison on 96 New Tasks. Non-Safe-Arrivals Receive 60 Seconds in Mean Completion Time; Audit Failure Is Not Necessarily a Physical Collision."
        table = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\caption{" + caption + r"}\label{tab:external}",
                 r"\begin{tabularx}{\textwidth}{@{}Xccccc@{}}\toprule", header + r"\\\midrule"]
        for method in ("egp", "dgppo"):
            for seed_index in (1, 2):
                selected = [row for row in groups if row["method"] == method and row["seed_index"] == seed_index]
                scale, cross = selected
                failed = sum(row["failed_audit_runs"] for row in selected)
                mean_time = sum(row["failure_penalized_completion"] * row["count"] for row in selected) / 96
                label = "EGP-NPA" if method == "egp" else ("DGPPO（任务适配）" if language == "zh" else "DGPPO (task-adapted)")
                table.append(f"{label} & {seed_index} & {scale['safe_arrivals']}/{scale['count']} & {cross['safe_arrivals']}/{cross['count']} & {failed}/96 & {mean_time:.2f} " + r"\\")
        table.extend([r"\bottomrule\end{tabularx}", r"\end{table*}"])
        (HERE / f"external_table_{language}.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
        descriptions = []
        for method in ("egp", "dgppo"):
            for seed_index in (1, 2):
                selected = [row for row in groups if row["method"] == method and row["seed_index"] == seed_index]
                arrivals = sum(row["arrived"] for row in selected)
                safe = sum(row["safe_arrivals"] for row in selected)
                collisions = sum(row["collision_runs"] for row in selected)
                label = "EGP-NPA" if method == "egp" else "DGPPO"
                if language == "zh":
                    descriptions.append(f"{label}第{seed_index}次训练的名义到达为${arrivals}/96$，安全到达为${safe}/96$，发生物理碰撞的运行有${collisions}$次。")
                else:
                    descriptions.append(f"For {label} replicate {seed_index}, nominal arrivals are ${arrivals}/96$, safe arrivals are ${safe}/96$, and ${collisions}$ runs have physical collisions.")
        (HERE / f"external_outcomes_{language}.tex").write_text("\n".join(descriptions) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
