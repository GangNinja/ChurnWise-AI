"""Consistency checks for the Phase 1 documentation set.

Run from the project root:

    python scripts/verify_phase1_docs.py

Exits 0 when every check passes, 1 otherwise. The script verifies that the
roadmap and acceptance criteria cover all thirteen phases, that every
acceptance phase documents its required subsections, that phase dependencies
point backwards only, that documentation links resolve, and that the mandatory
autonomous-data requirements are present.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED_FILES = [
    "ARCHITECTURE.md",
    "ROADMAP.md",
    "ACCEPTANCE_CRITERIA.md",
    "PHASE1_REPORT.md",
]

PHASE_TITLES = {
    1: "Repository, Architecture and Requirements",
    2: "Autonomous Dataset Discovery",
    3: "Autonomous Data Acquisition and Provenance",
    4: "Dataset Validation and Data Intelligence",
    5: "Preprocessing and Feature Engineering",
    6: "MLP Neural Network Development and Training",
    7: "Comparative Model Benchmarking and Evaluation",
    8: "Explainable AI and SHAP Integration",
    9: "Customer Churn Risk Intelligence",
    10: "Retention Campaign Simulation",
    11: "Interactive Intelligence Dashboard",
    12: "Integration, Reliability and End-to-End Testing",
    13: "Final Verification, Documentation and Presentation",
}

REQUIRED_SUBSECTIONS = [
    "Objectives and scope",
    "Inputs and outputs",
    "Expected artifacts",
    "Dependencies",
    "Testing and verification",
    "Completion criteria",
    "Risks and unresolved decisions",
]

PLACEHOLDER_PATTERN = re.compile(r"\b(TODO|TBD|FIXME|XXX|lorem ipsum)\b", re.IGNORECASE)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

failures: list[str] = []
passes: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    if ok:
        passes.append(label)
        print(f"PASS  {label}")
    else:
        failures.append(f"{label}: {detail}" if detail else label)
        print(f"FAIL  {label}{' — ' + detail if detail else ''}")


def slugify(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def headings(text: str) -> set[str]:
    return {slugify(m.group(2)) for m in HEADING_PATTERN.finditer(text)}


def sections(text: str, pattern: str) -> list[tuple[int, str]]:
    """Split text at each heading matching pattern; return (number, body)."""
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    out: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((int(m.group(1)), text[m.start():end]))
    return out


def main() -> int:
    print("ChurnWise AI — Phase 1 documentation verification\n")

    # 1. Required files exist.
    missing = [name for name in REQUIRED_FILES if not (DOCS / name).is_file()]
    check("required documentation files exist", not missing, f"missing {missing}")
    for extra in ("requirements.txt", ".gitignore"):
        check(f"supporting file {extra} exists", (ROOT / extra).is_file())
    if missing:
        print("\nCannot continue without all documents.")
        return 1

    texts = {name: (DOCS / name).read_text(encoding="utf-8") for name in REQUIRED_FILES}
    arch = texts["ARCHITECTURE.md"]
    roadmap = texts["ROADMAP.md"]
    accept = texts["ACCEPTANCE_CRITERIA.md"]

    # 2. Roadmap covers 13 phases with exact titles.
    roadmap_sections = sections(roadmap, r"^### Phase (\d+) — ")
    roadmap_phases = [num for num, _ in roadmap_sections]
    roadmap_titles = {
        int(m.group(1)): m.group(2).strip()
        for m in re.finditer(r"^### Phase (\d+) — (.+)$", roadmap, re.MULTILINE)
    }
    check("roadmap contains exactly 13 phase sections", len(roadmap_phases) == 13,
          f"found {len(roadmap_phases)}")
    check("roadmap phase numbers are 1..13", roadmap_phases == list(range(1, 14)),
          f"found {roadmap_phases}")
    title_mismatch = {
        n: (roadmap_titles.get(n), t) for n, t in PHASE_TITLES.items()
        if roadmap_titles.get(n, "").rstrip(".") != t
    }
    check("roadmap phase titles match the mandated phase list", not title_mismatch,
          f"{title_mismatch}")

    # 3. Acceptance criteria covers 13 phases, each with all subsections.
    accept_phases = sections(accept, r"^## \d+\.\s+Phase (\d+) — ")
    accept_nums = [n for n, _ in accept_phases]
    check("acceptance criteria contains exactly 13 phase sections", len(accept_phases) == 13,
          f"found {len(accept_phases)}")
    check("acceptance phase numbers are 1..13", accept_nums == list(range(1, 14)),
          f"found {accept_nums}")

    for num, body in accept_phases:
        absent = [s for s in REQUIRED_SUBSECTIONS if s.lower() not in body.lower()]
        check(f"phase {num} documents all seven required subsections", not absent, f"missing {absent}")
        boxes = len(re.findall(r"- \[[ xX]\]", body))
        check(f"phase {num} has explicit completion criteria (>=3)", boxes >= 3, f"found {boxes}")
        check(f"phase {num} declares at least one verification test",
              bool(re.search(r"\| T\d+\.\d+", body)), "no T<phase>.<n> test rows")

    # 4. Every roadmap phase declares dependencies; they point backwards only.
    for num, body in roadmap_sections:
        dep_match = re.search(r"\*\*Dependencies:\*\*\s*(.+)", body)
        check(f"roadmap phase {num} declares dependencies", dep_match is not None)
        if not dep_match:
            continue
        line = dep_match.group(1)
        if num == 1:
            check("phase 1 has no upstream dependency", "none" in line.lower(), line.strip())
            continue
        deps = {int(d) for d in re.findall(r"(?:P|Phase\s+)(\d+)", line)}
        check(f"phase {num} dependencies are non-empty", bool(deps), line.strip())
        forward = sorted(d for d in deps if d >= num)
        check(f"phase {num} has only backward dependencies", not forward, f"forward refs {forward}")

    # 4b. Schedule arithmetic: per-phase and per-day hours must reconcile.
    effort = re.search(r"\*\*Effort allocation \(([\d.]+) h total\):\*\*(.+)", roadmap)
    check("roadmap declares an effort allocation total", effort is not None)
    if effort:
        declared_total = float(effort.group(1))
        effort_hours = {
            int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"P(\d+)\s+([\d.]+)\s*h", effort.group(2))
        }
        scheduled: dict[int, float] = {}
        for m in re.finditer(r"P(\d+)\s*\(([\d.]+)\s*h\)", roadmap):
            num, hours = int(m.group(1)), float(m.group(2))
            scheduled[num] = scheduled.get(num, 0.0) + hours
        check("effort allocation covers phases 1..13", set(effort_hours) == set(range(1, 14)),
              f"found {sorted(effort_hours)}")
        check("scheduled hours cover phases 1..13", set(scheduled) == set(range(1, 14)),
              f"found {sorted(scheduled)}")
        mismatch = {
            n: (effort_hours.get(n), scheduled.get(n))
            for n in range(1, 14)
            if abs(effort_hours.get(n, 0.0) - scheduled.get(n, 0.0)) > 1e-9
        }
        check("per-phase scheduled hours equal the effort allocation", not mismatch, f"{mismatch}")
        check("phase hours sum to the declared total",
              abs(sum(effort_hours.values()) - declared_total) < 1e-9,
              f"{sum(effort_hours.values())} vs {declared_total}")
        day_hours = [
            float(m.group(1))
            for m in re.finditer(r"^\|\s*\d\s*\|\s*([\d.]+)\s*\|", roadmap, re.MULTILINE)
        ]
        check("schedule lists five working days", len(day_hours) == 5, f"found {day_hours}")
        check("daily planned hours sum to the declared total",
              abs(sum(day_hours) - declared_total) < 1e-9,
              f"{sum(day_hours)} vs {declared_total}")

    # 5. Diagrams.
    for name, text in (("ARCHITECTURE.md", arch), ("ROADMAP.md", roadmap)):
        check(f"{name} contains a Mermaid diagram", "```mermaid" in text)

    # 6. Mandatory autonomy requirements.
    autonomy_phrases = {
        "ARCHITECTURE.md": ["without supplying a file or a URL", "mandatory"],
        "ROADMAP.md": ["remain mandatory", "never supplies a dataset", "autonomous dataset discovery"],
        "ACCEPTANCE_CRITERIA.md": ["mandatory", "without any user-supplied input", "synthetic"],
    }
    for name, phrases in autonomy_phrases.items():
        text = texts[name].lower()
        absent = [p for p in phrases if p.lower() not in text]
        check(f"{name} mandates autonomous data acquisition", not absent, f"missing {absent}")
    check("architecture forbids synthetic fallback data",
          "prohibited" in arch.lower() and "synthetic" in arch.lower())
    check("acceptance criteria requires a no-synthetic-data test",
          "no synthetic" in accept.lower())

    # 7. ML requirements coverage.
    ml_terms = ["relu", "sigmoid", "binary cross-entropy", "adam", "backpropagation",
                "early stopping", "dropout", "l2"]
    absent = [t for t in ml_terms if t not in arch.lower()]
    check("architecture specifies the full MLP training design", not absent, f"missing {absent}")
    metrics = ["accuracy", "precision", "recall", "f1", "roc-auc", "confusion matri"]
    absent = [m for m in metrics if m not in accept.lower()]
    check("acceptance criteria require all mandatory metrics", not absent, f"missing {absent}")
    check("acceptance criteria requires held-out test evidence",
          "held-out test" in accept.lower())

    # 8. No predetermined performance targets asserted as expectations.
    check("no fabricated performance target is stated",
          not re.search(r"target\s+(roc|auc|accuracy)\s*[=:>]", arch + roadmap + accept, re.I))

    # 9. Scope preservation and definition of done.
    check("roadmap states scope preservation", "scope preservation" in roadmap.lower())
    check("acceptance criteria defines a global definition of done",
          "definition of done" in accept.lower())

    # 10. Placeholders (ignore inline code, where forbidden markers are named).
    for name, text in texts.items():
        prose = re.sub(r"`[^`]*`", "", text)
        hits = PLACEHOLDER_PATTERN.findall(prose)
        check(f"{name} contains no placeholder tokens", not hits, f"found {set(hits)}")

    # 11. Links resolve (path and anchor).
    for name, text in texts.items():
        broken = []
        for label, target in LINK_PATTERN.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part, _, anchor = target.partition("#")
            resolved = (DOCS / path_part).resolve()
            if not (resolved.is_file() or resolved.is_dir()):
                broken.append(f"{label} -> {target}")
                continue
            if anchor:
                target_text = resolved.read_text(encoding="utf-8")
                if anchor not in headings(target_text):
                    broken.append(f"{label} -> {target} (anchor)")
        check(f"{name} internal links resolve", not broken, f"{broken}")

    # 12. Traceability matrix covers every phase.
    trace_rows = {
        int(m.group(1))
        for m in re.finditer(r"^\|\s*(\d+)\s*\|", accept, re.MULTILINE)
    }
    check("traceability matrix covers phases 1..13", set(range(1, 14)).issubset(trace_rows),
          f"missing {sorted(set(range(1, 14)) - trace_rows)}")

    print(f"\n{len(passes)} checks passed, {len(failures)} failed")
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
