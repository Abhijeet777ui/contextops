from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
import dataclasses
from .models import ContextBenchSample

class BaseGenerator(ABC):
    def __init__(self, version: str = "schema_v2"):
        self.version = version

    @abstractmethod
    def generate(self, count: int, domain: str, difficulty: str, **kwargs) -> List[ContextBenchSample]:
        """
        Generate a list of benchmark samples.
        Must be implemented by specific generators (e.g., HealthyRAGGenerator, StructuralFailureGenerator).
        """
        pass

    def save_dataset(self, samples: List[ContextBenchSample], output_file: str):
        """Save samples to a JSONL file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample in samples:
                # Convert dataclass to dict and write as JSON Lines
                sample_dict = dataclasses.asdict(sample)
                f.write(json.dumps(sample_dict) + '\n')
