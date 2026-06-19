"""
TemporalDriftGenerator — Phase 5: Temporal Context Drift

Three failure subtypes:
  1. stale_memory_injection — injecting memory entries that have been invalidated by recent events
  2. retrieval_drift — retrieving outdated v1 docs alongside v2/v3 docs, creating contradictory context
  3. summary_compression_drift — meaning degrades over repeated memory summarization loops
"""
import random
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
from .domain_templates import SYSTEM_PROMPTS, QUERIES, RETRIEVAL_CHUNKS

# ---------------------------------------------------------------------------
# Contradictory Document Versions (Retrieval Drift)
# Simulates retrieving both old and new versions of the same document
# ---------------------------------------------------------------------------
DOCUMENT_VERSIONS = {
    "legal": [
        ("v1", "Section 14 (Limitation of Liability): The aggregate liability of either party shall not exceed $100,000 in any given calendar year. (Dated: Jan 2023)"),
        ("v2", "Section 14 (Limitation of Liability): The aggregate liability of either party shall not exceed the total fees paid in the twelve (12) months preceding the claim. (Dated: Jun 2024)"),
    ],
    "finance": [
        ("v1", "Q3 Revenue Guidance: Management expects Q3 FY2024 revenue to be in the range of $82-84 billion, citing macroeconomic headwinds. (Issued: April 2024)"),
        ("v2", "Q3 Revenue Actual: Apple reported Q3 FY2024 revenue of $85.8 billion, exceeding the high end of previous guidance. (Issued: August 2024)"),
    ],
    "medical": [
        ("v1", "Allergies: None known. (Recorded: 2022-01-15)"),
        ("v2", "Allergies: Penicillin (rash), Sulfonamides (anaphylaxis). (Recorded: 2024-06-10)"),
    ],
    "code": [
        ("v1", "API Documentation: The endpoint `/api/v1/users` accepts a standard API key in the `X-API-Key` header. Rate limit: 1000 req/min. (Deprecation warning: 2023-11)"),
        ("v2", "API Documentation: The endpoint `/api/v2/users` requires an OAuth 2.0 Bearer token. `X-API-Key` is no longer supported. Rate limit: 5000 req/min. (Active)"),
    ],
    "research": [
        ("v1", "Pre-print findings: We demonstrate that increasing context window beyond 32k tokens inevitably degrades retrieval accuracy by 40% (lost-in-the-middle effect). (ArXiv, v1)"),
        ("v2", "Published findings: By implementing RoPE scaling and cross-layer attention, we maintain 98% retrieval accuracy up to 128k tokens, resolving the degradation seen in early experiments. (NeurIPS, final)"),
    ],
    "customer_support": [
        ("v1", "Policy: Customers on the Starter plan are eligible for phone support between 9am-5pm EST. (Valid through 2023)"),
        ("v2", "Policy: Customers on the Starter plan have access to email support only. Phone support is restricted to Business and Enterprise tiers. (Effective Jan 1, 2024)"),
    ],
    "enterprise_docs": [
        ("v1", "Runbook: Deployments to production must be triggered manually via Jenkins. (Last updated: 2022)"),
        ("v2", "Runbook: All production deployments are fully automated via GitHub Actions upon merging to the `main` branch. Manual Jenkins deployments are disabled. (Last updated: 2024)"),
    ],
}

# ---------------------------------------------------------------------------
# Summary Compression Degradation
# Simulates an agent that summarizes its own summaries repeatedly until
# critical details (like 'do not restart') are lost or inverted.
# ---------------------------------------------------------------------------
COMPRESSION_DEGRADATION = {
    "code": [
        "Turn 1: The database is under heavy load. A restart will drop all active connections and cause data loss during the migration. DO NOT RESTART. We must wait for the migration to finish.",
        "Turn 2 (Summary): Database is heavily loaded due to migration. Restarting will cause data loss. We are waiting.",
        "Turn 3 (Summary): DB load is high from migration. Risk of data loss if restarted.",
        "Turn 4 (Summary): DB migration causing high load. Restart is related to data loss.",
        "Turn 5 (Summary): High DB load. Consider restart to resolve migration data issues.",
    ],
    "medical": [
        "Turn 1: Patient presents with severe headache. Note: patient has a severe allergy to NSAIDs (anaphylaxis risk). Recommend Acetaminophen 1000mg. DO NOT PRESCRIBE IBUPROFEN.",
        "Turn 2 (Summary): Severe headache. Allergy to NSAIDs. Recommend Acetaminophen. Ibuprofen contraindicated.",
        "Turn 3 (Summary): Headache. NSAID allergy noted. Acetaminophen recommended over Ibuprofen.",
        "Turn 4 (Summary): Headache treatment options discussed include NSAIDs and Acetaminophen.",
        "Turn 5 (Summary): Patient with headache. Recommend standard NSAID treatment like Ibuprofen or Acetaminophen.",
    ],
    "legal": [
        "Turn 1: The non-compete clause is explicitly carved out and DOES NOT apply to employees in California due to state law restrictions.",
        "Turn 2 (Summary): Non-compete clause does not apply in California due to legal restrictions.",
        "Turn 3 (Summary): Non-compete clause restricted by California state law.",
        "Turn 4 (Summary): California law governs the non-compete clause restrictions.",
        "Turn 5 (Summary): The non-compete clause applies under California law.",
    ]
}

# ---------------------------------------------------------------------------
# Stale Memory States
# ---------------------------------------------------------------------------
STALE_STATES = {
    "customer_support": [
        "[2023-11] Customer Plan: Starter ($49/mo).",
        "[2024-02] Customer upgraded to Business plan ($299/mo).",
        "[2024-05] Customer added 50 extra seats to Business plan.",
    ],
    "enterprise_docs": [
        "[10:00 AM] Incident P1 started. Primary DB is down.",
        "[10:15 AM] Failover to secondary DB successful. Read-only mode.",
        "[11:30 AM] Primary DB restored. Full read/write functionality restored. Incident closed.",
    ],
    "finance": [
        "[Q1] User holds 500 shares of TSLA.",
        "[Q2] User sold all TSLA shares and bought 1000 shares of NVDA.",
    ]
}


class TemporalDriftGenerator(BaseGenerator):
    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "temporal_drift"
    GENERATOR_NAME = "temporal_drift_gen_v1"

    def generate(
        self,
        count: int,
        domain: str = "all",
        difficulty: str = "medium",
        subtype: str = "all",
        **kwargs,
    ) -> List[ContextBenchSample]:
        domains = list(SYSTEM_PROMPTS.keys()) if domain == "all" else [domain]
        subtypes = (
            ["retrieval_drift", "summary_compression_drift", "stale_memory_injection"]
            if subtype == "all"
            else [subtype]
        )
        samples: List[ContextBenchSample] = []

        for i in range(count):
            current_domain = domains[i % len(domains)]
            current_subtype = subtypes[i % len(subtypes)]
            sample = self._build_sample(i, current_domain, difficulty, current_subtype)
            samples.append(sample)

        return samples

    def _build_sample(
        self, index: int, domain: str, difficulty: str, subtype: str
    ) -> ContextBenchSample:
        if subtype == "retrieval_drift":
            return self._retrieval_drift(index, domain, difficulty)
        elif subtype == "summary_compression_drift":
            return self._summary_compression_drift(index, domain, difficulty)
        else:
            return self._stale_memory_injection(index, domain, difficulty)

    def _retrieval_drift(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Retrieve both v1 and v2 of a document, creating a contradiction
        versions = DOCUMENT_VERSIONS.get(domain, DOCUMENT_VERSIONS["finance"])
        v1_text = versions[0][1]
        v2_text = versions[1][1]

        # Add some healthy chunks as well
        available = RETRIEVAL_CHUNKS[domain]
        chunks = random.sample(available, min(2, len(available)))
        chunks.extend([v1_text, v2_text])
        random.shuffle(chunks)

        query = random.choice(QUERIES[domain])
        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=[],
            memory=[],
        )

        approx_tokens = (len(SYSTEM_PROMPTS[domain]) + len(query) + sum(len(c) for c in chunks)) // 4

        ground_truth = GroundTruth(
            failure_modes=["retrieval_drift", "context_contradiction"],
            severity="high",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=False,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )

        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "retrieval_drift", "versions_included": ["v1", "v2"]},
        )

        return ContextBenchSample(
            id=f"cb_v1_temp_retrieval_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="retrieval_drift",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    def _summary_compression_drift(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Provide the degraded summary as the only memory entry
        # Fallback to code domain if current domain not in dict
        compression_chain = COMPRESSION_DEGRADATION.get(domain, COMPRESSION_DEGRADATION["code"])
        
        # Hard difficulty uses the fully degraded (inverted) memory
        # Easy uses slightly degraded
        level = {"easy": 2, "medium": 3, "hard": 4}.get(difficulty, 3)
        degraded_memory = compression_chain[level]

        query = random.choice(QUERIES[domain])
        chunks = random.sample(RETRIEVAL_CHUNKS[domain], min(2, len(RETRIEVAL_CHUNKS[domain])))

        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=[],
            memory=[f"Agent Summary: {degraded_memory}"],
        )

        approx_tokens = (len(SYSTEM_PROMPTS[domain]) + len(query) + sum(len(c) for c in chunks) + len(degraded_memory)) // 4

        ground_truth = GroundTruth(
            failure_modes=["summary_compression_drift", "memory_inversion"],
            severity="critical" if level == 4 else "high",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=False,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )

        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "summary_compression_drift", "compression_turns": level},
        )

        return ContextBenchSample(
            id=f"cb_v1_temp_summary_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="summary_compression_drift",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    def _stale_memory_injection(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Inject old, invalidated state alongside new state
        states = STALE_STATES.get(domain, STALE_STATES["enterprise_docs"])
        
        # We inject the V1 state (stale) and V3 state (current) into memory
        memory = [states[0], states[-1]]

        query = random.choice(QUERIES[domain])
        chunks = random.sample(RETRIEVAL_CHUNKS[domain], min(2, len(RETRIEVAL_CHUNKS[domain])))

        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=[],
            memory=memory,
        )

        approx_tokens = (len(SYSTEM_PROMPTS[domain]) + len(query) + sum(len(c) for c in chunks) + sum(len(m) for m in memory)) // 4

        ground_truth = GroundTruth(
            failure_modes=["stale_memory_injection", "state_contradiction"],
            severity="medium",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=False,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )

        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "stale_memory_injection", "stale_entries": 1, "valid_entries": 1},
        )

        return ContextBenchSample(
            id=f"cb_v1_temp_stale_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="stale_memory_injection",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )
