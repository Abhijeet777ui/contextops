from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ContextData:
    system_prompt: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    retrieval_chunks: List[str] = field(default_factory=list)
    tool_outputs: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)

@dataclass
class ExpectedProperties:
    contains_redundancy: bool = False
    contains_density_bloat: bool = False
    contains_structure_imbalance: bool = False
    contains_source_concentration: bool = False

@dataclass
class GroundTruth:
    failure_modes: List[str]
    severity: str  # e.g., "healthy", "low", "medium", "critical"
    architecture_failure: bool
    expected_properties: ExpectedProperties

@dataclass
class MutationMetadata:
    generator: str
    parameters: Dict[str, Any]

@dataclass
class BenchmarkMetadata:
    token_count: int
    version: str

@dataclass
class ContextBenchSample:
    id: str
    benchmark: str
    category: str
    subcategory: str
    domain: str
    difficulty: str
    context: ContextData
    ground_truth: GroundTruth
    mutation: MutationMetadata
    metadata: BenchmarkMetadata
