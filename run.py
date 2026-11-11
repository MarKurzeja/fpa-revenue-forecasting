"""Command-line entry point (SPEC §13). Run `python run.py --help`."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fpa.config import load_config  # noqa: E402
from fpa.pipeline import timed  # noqa: E402

STEPS = ("build", "backtest", "reconcile", "plan", "capital", "report", "export")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retail revenue forecasting for FP&A (Walmart M5)")
    ap.add_argument("command", choices=["download", *STEPS, "app", "all", "check"])
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument(
        "--quick",
        action="store_true",
        help="fast real-data run (3 stores, fewer origins) as used by `check`",
    )
    ap.add_argument("--force", action="store_true", help="recompute cached results")
    ap.add_argument("--models", default="all", choices=["all", "baselines", "stats", "ml"])
    ap.add_argument("--origins", type=int, default=None, help="override number of origins")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "check":
        return check(args)
    cfg = load_config(args.config, quick=args.quick)
    if args.origins:
        import dataclasses

        cfg = dataclasses.replace(
            cfg,
            backtest=dataclasses.replace(
                cfg.backtest,
                n_origins=args.origins,
                n_validation_origins=min(cfg.backtest.n_validation_origins, args.origins - 1),
            ),
        )
    cfg.paths.ensure()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    if args.command == "download":
        from fpa.data.download import download

        return 0 if download(cfg.paths.raw_dir, cfg.data["kaggle_competition"]) else 1
    if args.command == "app":
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")]
        )
    steps = STEPS if args.command == "all" else (args.command,)
    for step in steps:
        with timed(step, timings):
            run_step(step, cfg, args)
    print(
        "\nRuntime per step: "
        + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        + f"  (total {time.perf_counter() - t0:.1f}s)"
    )
    return 0


def run_step(step: str, cfg, args) -> None:
    from fpa import pipeline

    if step == "build":
        pipeline.step_build(cfg, force=args.force)
    elif step == "backtest":
        pipeline.step_backtest(cfg, force=args.force, model_set=args.models)
    elif step == "reconcile":
        pipeline.step_reconcile(cfg)
    elif step == "plan":
        pipeline.step_plan(cfg)
    elif step == "capital":
        pipeline.step_capital(cfg)
    elif step == "report":
        from fpa.report import build_report

        print(f"    report -> {build_report(cfg)}")
    elif step == "export":
        from fpa.export import export_demo, export_powerbi

        print(f"    Power BI CSVs -> {export_powerbi(cfg)}")
        print(f"    app artefacts -> {export_demo(cfg)}")


def check(args) -> int:
    """Lint + tests + quick real-data pipeline with quality gates (SPEC §14.2, D-13)."""
    rc = 0
    for cmd in (
        ["ruff", "check", "."],
        ["ruff", "format", "--check", "."],
        [sys.executable, "-m", "pytest", "--cov=fpa", "--cov-report=term-missing:skip-covered"],
    ):
        print("$ " + " ".join(cmd))
        rc |= subprocess.call(cmd, cwd=ROOT)
    if rc:
        print("lint/tests failed")
        return rc
    from fpa.data.download import has_raw_data

    cfg = load_config(args.config, quick=True)
    if not has_raw_data(cfg.paths.raw_dir):
        print("M5 data not found: skipping the pipeline gate (run `python run.py download`).")
        return 0
    rc = main(["all", "--quick", *(["--force"] if args.force else [])])
    if rc:
        return rc
    from fpa.pipeline import quality_gates

    fails = quality_gates(cfg)
    for f in fails:
        print("QUALITY GATE FAILED: " + f)
    print("check passed" if not fails else "check failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
