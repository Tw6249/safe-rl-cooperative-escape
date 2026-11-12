import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = HERE.parents[1]


def main():
    sources = json.loads((HERE / "results_matched" / "source_sha256.json").read_text(encoding="utf-8"))
    for relative, expected in sources.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source changed: {relative}")
    summary = json.loads((HERE / "results_matched" / "summary.json").read_text(encoding="utf-8"))
    if sum(row["count"] for row in summary["rows"]) != 384:
        raise ValueError("Missing matched evaluations")
    for language in ("zh", "en"):
        main_path = PAPER / f"main_{language}.tex"
        main_text = main_path.read_text(encoding="utf-8")
        inputs = re.findall(r"\\input\{([^}]+)\}", main_text)
        active = main_text + "\n" + "\n".join((PAPER / f"{name}.tex").read_text(encoding="utf-8") for name in inputs)
        labels = re.findall(r"\\label\{([^}]+)\}", active)
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate labels: {language}")
        for forbidden in ("Static-CBF GNN", "GCBF+/DGPPO paradigm", "liu2026saferl", "53.1\\%", "必然发生安全违约"):
            if forbidden in active:
                raise ValueError(f"Misleading historical claim remains active: {language}/{forbidden}")
        experiment = (PAPER / f"sections_{language}" / "05_experiments_revised.tex").read_text(encoding="utf-8")
        for critic in ("team", "local"):
            for penalty in (0, 1):
                for row in summary["rows"]:
                    if row["critic"] == critic and row["projection_weight"] == penalty:
                        if f"{row['safe_arrivals']}/{row['count']}" not in experiment:
                            raise ValueError("Matched table has a missing count")
        log = (PAPER / f"build_{language}" / f"main_{language}.log").read_text(encoding="utf-8", errors="replace")
        if "Overfull \\hbox" in log or re.search(r"LaTeX Warning:|Package .* Warning:", log):
            raise ValueError(f"Compilation warning: {language}")
    corrected = json.loads((HERE / "results_exploratory_corrected" / "summary.json").read_text(encoding="utf-8"))
    if len(corrected) != 96 or any(row["holding_duration"] < 1 - 1e-9 for row in corrected):
        raise ValueError("Corrected evaluation holding criterion failed")
    report = dict(source_hashes_checked=len(sources), matched_evaluations=384, corrected_evaluations=len(corrected),
                  corrected_safe_arrivals=sum(row["safe_arrived"] for row in corrected),
                  corrected_failed_audit_segments=sum(row["audit_failed"] for row in corrected),
                  active_manuscripts_checked=2, compilation_warnings=0)
    (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
