"""
STEP 14: Full pipeline runner
Run this to execute the complete training pipeline from scratch,
or select specific steps to run.

Usage:
  python step14_run_pipeline.py              # Run all steps
  python step14_run_pipeline.py --steps 1 2 3 4 5 6 7
  python step14_run_pipeline.py --from 5     # Run from step 5 onwards
  python step14_run_pipeline.py --steps 10   # Just retrain
"""

import argparse
import subprocess
import sys
import time
import os

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    1:  ("Clean datasets",        "step1_clean_datasets.py"),
    2:  ("Verify class config",   "step2_class_config.py"),
    3:  ("Remap labels",          "step3_convert_labels.py"),
    4:  ("Merge datasets",        "step4_merge_datasets.py"),
    5:  ("Split train/val",       "step5_split_dataset.py"),
    6:  ("Create data.yaml",      "step6_create_yaml.py"),
    7:  ("Train model",           "step7_train.py"),
    8:  ("Evaluate",              "step8_evaluate.py"),
    9:  ("Analyze dataset",       "step9_improve_dataset.py"),
    10: ("Retrain",               "step10_retrain.py"),
    11: ("Export model",          "step11_export_model.py"),
    # Steps 12, 13 are integration/test scripts, not pipeline steps
}


def run_step(step_num, script_name):
    script_path = os.path.join(PIPELINE_DIR, script_name)
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {STEPS[step_num][0]}")
    print(f"  Script: {script_name}")
    print(f"{'='*70}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(PIPELINE_DIR),  # Run from project root
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] Step {step_num} failed after {elapsed:.1f}s")
        return False

    print(f"\n[OK] Step {step_num} completed in {elapsed:.1f}s")
    return True


def run_pipeline(steps_to_run):
    print("\n" + "="*70)
    print("  JalDrishti Marine Detection - Training Pipeline")
    print("="*70)
    print(f"  Steps to run: {steps_to_run}")

    pipeline_start = time.time()
    failed_at = None

    for step in steps_to_run:
        if step not in STEPS:
            print(f"[WARNING] Step {step} not found in pipeline. Skipping.")
            continue
        _, script = STEPS[step]
        success = run_step(step, script)
        if not success:
            failed_at = step
            break

    total_time = time.time() - pipeline_start
    print(f"\n{'='*70}")
    if failed_at:
        print(f"  PIPELINE FAILED at Step {failed_at}  ({total_time:.0f}s total)")
    else:
        print(f"  PIPELINE COMPLETE  ({total_time:.0f}s total)")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marine Detection Pipeline Runner")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        help="Specific step numbers to run (e.g. --steps 1 2 3)"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int,
        help="Run from this step to the end (e.g. --from 5)"
    )
    args = parser.parse_args()

    all_steps = sorted(STEPS.keys())

    if args.steps:
        steps = sorted(args.steps)
    elif args.from_step:
        steps = [s for s in all_steps if s >= args.from_step]
    else:
        steps = all_steps

    run_pipeline(steps)
