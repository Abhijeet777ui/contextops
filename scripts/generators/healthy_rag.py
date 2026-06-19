"""
HealthyRAGGenerator — Phase 2: Healthy Architectures

Generates benchmark samples that represent CORRECTLY architected RAG pipelines.
These are positive (non-failing) samples — essential to prevent benchmark bias
where an evaluator could score perfectly by always predicting failure.

Healthy RAG characteristics:
- Concise, focused system prompt (no bloat)
- 3-5 distinct, non-overlapping retrieval chunks from diverse sources
- Compressed, relevant memory (0-1 entries max)
- Optional lean tool output (single paginated result)
- No redundancy, no flooding, no injection
"""
import random
import uuid
from typing import List

from .base import BaseGenerator
from .models import (
    BenchmarkMetadata,
    ContextBenchSample,
    ContextData,
    ExpectedProperties,
    GroundTruth,
    MutationMetadata,
)
from .domain_templates import (
    SYSTEM_PROMPTS,
    QUERIES,
    RETRIEVAL_CHUNKS,
    MEMORY_SNIPPETS,
    TOOL_OUTPUTS,
)


class HealthyRAGGenerator(BaseGenerator):
    """
    Generates healthy, well-architected RAG samples.
    These are the positive class — contexts that an evaluator should
    NOT flag as failures.
    """

    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "healthy_architectures"
    SUBCATEGORY = "efficient_rag"
    GENERATOR_NAME = "healthy_rag_gen_v1"

    # How many chunks a healthy RAG should have (top-K discipline)
    MIN_CHUNKS = 3
    MAX_CHUNKS = 5

    def generate(
        self,
        count: int,
        domain: str = "all",
        difficulty: str = "easy",
        include_memory: bool = True,
        include_tool_output: bool = False,
    ) -> List[ContextBenchSample]:
        """
        Generate healthy RAG samples.

        Args:
            count: Number of samples to generate.
            domain: Target domain (e.g. 'legal', 'finance') or 'all' to cycle.
            difficulty: 'easy' | 'medium'. Easy = no memory, no tools.
            include_memory: If True, optionally include one relevant memory entry.
            include_tool_output: If True, optionally include one lean tool output.

        Returns:
            List of ContextBenchSample objects.
        """
        domains = list(SYSTEM_PROMPTS.keys()) if domain == "all" else [domain]
        samples: List[ContextBenchSample] = []

        for i in range(count):
            current_domain = domains[i % len(domains)]
            sample = self._build_sample(
                index=i,
                domain=current_domain,
                difficulty=difficulty,
                include_memory=include_memory,
                include_tool_output=include_tool_output,
            )
            samples.append(sample)

        return samples

    def _build_sample(
        self,
        index: int,
        domain: str,
        difficulty: str,
        include_memory: bool,
        include_tool_output: bool,
    ) -> ContextBenchSample:
        system_prompt = SYSTEM_PROMPTS[domain]
        query = random.choice(QUERIES[domain])

        # Select 3-5 non-repeating diverse chunks
        available_chunks = RETRIEVAL_CHUNKS[domain]
        num_chunks = random.randint(self.MIN_CHUNKS, min(self.MAX_CHUNKS, len(available_chunks)))
        chosen_chunks = random.sample(available_chunks, num_chunks)

        # Optionally add a single memory entry (healthy = at most 1, recent, relevant)
        memory = []
        if include_memory and difficulty != "easy" and random.random() > 0.4:
            memory = [random.choice(MEMORY_SNIPPETS[domain])]

        # Optionally add one lean, paginated tool output
        tool_outputs = []
        if include_tool_output and random.random() > 0.5:
            tool_outputs = [TOOL_OUTPUTS[domain][0]]

        context = ContextData(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chosen_chunks,
            tool_outputs=tool_outputs,
            memory=memory,
        )

        # Approximate token count (rough: 1 token ≈ 4 chars)
        total_text = (
            system_prompt
            + query
            + " ".join(chosen_chunks)
            + " ".join(memory)
            + " ".join(tool_outputs)
        )
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=[],
            severity="healthy",
            architecture_failure=False,
            expected_properties=ExpectedProperties(
                contains_redundancy=False,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )

        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={
                "num_chunks": num_chunks,
                "include_memory": bool(memory),
                "include_tool_output": bool(tool_outputs),
                "chunk_sampling": "random_no_replacement",
            },
        )

        metadata = BenchmarkMetadata(
            token_count=approx_tokens,
            version=self.version,
        )

        sample_id = f"cb_v1_healthy_rag_{index:04d}_{domain}"

        return ContextBenchSample(
            id=sample_id,
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory=self.SUBCATEGORY,
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=metadata,
        )
