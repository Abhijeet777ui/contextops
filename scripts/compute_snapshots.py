"""Compute current benchmark snapshot values for test_benchmarks.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextops.api.inspect import inspect_context

SNAPSHOTS = [
    "01_simple_redundancy.json",
    "02_semantic_duplicates.json",
    "03_rag_overload.json",
    "04_agent_trace_noise.json",
    "05_structured_data_case.json",
    "06_fake_good_context.json",
    "07_instruction_conflict.json",
]

benchmarks_dir = Path("benchmarks")
print("SNAPSHOTS = {")
for file_name in SNAPSHOTS:
    file_path = benchmarks_dir / file_name
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    context_data = data.get("input", data)
    result = inspect_context(context_data)
    top_rec = result.recommendations[0].issue if result.recommendations else None
    print(f'    "{file_name}": {{')
    print(f'        "score": {result.score},')
    print(f'        "redundancy_penalty": {result.score_breakdown.redundancy_penalty},')
    print(f'        "structure_penalty": {result.score_breakdown.structure_penalty},')
    print(f'        "top_recommendation_issue": {json.dumps(top_rec)}')
    print("    },")
print("}")
