"""
Master Runner for ContextBench and ContextSecBench

This script sequentially executes all generation phases to build the complete
benchmark suites from scratch.

Run from the project root:
    python scripts/generate_all.py
"""
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_phase2_healthy import main as gen_phase2
from scripts.generate_phase3_failures import main as gen_phase3
from scripts.generate_phase4_sec import main as gen_phase4
from scripts.generate_phase5_temporal import main as gen_phase5

def main():
    print("=========================================================")
    print("  CONTEXTBENCH & CONTEXTSECBENCH GENERATION SUITE v1.0   ")
    print("=========================================================\n")

    print(">>> PHASE 2: Generating Healthy Architectures (300 samples)")
    gen_phase2()
    print("-" * 50 + "\n")

    print(">>> PHASE 3: Generating Architecture Failures (900 samples)")
    gen_phase3()
    print("-" * 50 + "\n")

    print(">>> PHASE 4: Generating ContextSecBench (300 samples)")
    gen_phase4()
    print("-" * 50 + "\n")

    print(">>> PHASE 5: Generating Temporal Drift Failures (300 samples)")
    gen_phase5()
    print("-" * 50 + "\n")

    print("=========================================================")
    print(" ALL GENERATION PHASES COMPLETE.")
    print(" Total Samples Generated: 1800")
    print(" Data output to ContextBench/ and ContextSecBench/ directories.")
    print("=========================================================")

if __name__ == "__main__":
    main()
