"""
HealthyMultiAgentGenerator — Phase 2: Healthy Architectures

Generates benchmark samples that represent well-architected multi-agent pipelines.
A healthy multi-agent context has:
- Clean agent handoffs (Planner → Executor → Validator pattern)
- Compressed summaries — not raw repeated outputs
- No memory accumulation from stale prior turns
- Lean, purpose-specific system prompt per agent role
- No cross-agent context duplication
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
from .domain_templates import (
    SYSTEM_PROMPTS,
    QUERIES,
    RETRIEVAL_CHUNKS,
    MEMORY_SNIPPETS,
    AGENT_ROLES,
)


# Realistic planner outputs — short, structured task breakdowns
PLANNER_OUTPUTS = {
    "legal": [
        "Plan: (1) Retrieve indemnification clauses from contract. (2) Identify liability caps. (3) Cross-reference with governing law section. (4) Summarize obligations for both parties.",
        "Plan: (1) Pull all termination clauses. (2) Identify cure periods. (3) Note any unilateral termination rights. (4) Flag IP ownership implications on termination.",
    ],
    "finance": [
        "Plan: (1) Retrieve Q3 revenue and margin figures. (2) Calculate YoY delta. (3) Extract management guidance. (4) Summarize capital allocation highlights.",
        "Plan: (1) Pull segment-level revenue data. (2) Identify fastest and slowest growing segments. (3) Extract CFO commentary on drivers. (4) Compile into executive summary.",
    ],
    "medical": [
        "Plan: (1) Retrieve current medication list. (2) Check for drug interactions. (3) Review latest lab results. (4) Summarize for physician review.",
        "Plan: (1) Pull active problem list with ICD-10 codes. (2) Review HbA1c trend. (3) Flag overdue screenings. (4) Draft clinical note.",
    ],
    "code": [
        "Plan: (1) Parse stack trace to identify error origin. (2) Locate relevant code section. (3) Identify root cause. (4) Propose fix with explanation.",
        "Plan: (1) Analyze query execution plan. (2) Identify missing indexes. (3) Suggest schema changes. (4) Estimate performance improvement.",
    ],
    "research": [
        "Plan: (1) Retrieve key papers on the topic. (2) Extract methodology from each. (3) Compare findings across papers. (4) Identify gaps and conflicts.",
        "Plan: (1) Pull ablation study results. (2) Rank contributions by magnitude. (3) Compare to baseline. (4) Summarize for literature review.",
    ],
    "customer_support": [
        "Plan: (1) Look up customer account and recent charges. (2) Identify duplicate charge. (3) Initiate refund process. (4) Draft customer communication.",
        "Plan: (1) Check SSO configuration for the customer's domain. (2) Identify misconfiguration. (3) Provide resolution steps. (4) Verify fix applied.",
    ],
    "enterprise_docs": [
        "Plan: (1) Retrieve incident escalation policy. (2) Identify P0 criteria. (3) Pull on-call contacts. (4) Draft incident response checklist.",
        "Plan: (1) Retrieve Kafka runbook. (2) Check current consumer lag. (3) Determine restart is safe. (4) Document steps taken.",
    ],
}

# Realistic executor outputs — concise, factual, no bloat
EXECUTOR_OUTPUTS = {
    "legal": [
        "Executed: Retrieved Sections 12.1, 12.2, and 14. Key findings: Service provider indemnifies for IP infringement; liability capped at 12 months of fees paid.",
        "Executed: Identified Section 9.3 (Termination for Cause) and Section 9.4 (Termination for Convenience). Cure period: 30 days. 90-day notice required for convenience termination.",
    ],
    "finance": [
        "Executed: Q3 FY2024 revenue = $85.8B (+5% YoY). Gross margin = 46.3% (+1.8pp YoY). Services = $24.2B (+14% YoY). Q4 guidance: $89-91B.",
        "Executed: Highest growth segment = Services (+14%). Lowest = Wearables (-9%). CFO cited installed base growth and AI features as primary revenue drivers.",
    ],
    "medical": [
        "Executed: Current meds — Metformin 1000mg BID, Lisinopril 10mg QD, Atorvastatin 40mg QHS. No critical drug-drug interactions at current doses. Allergy flag: Penicillin (rash), Sulfonamides (anaphylaxis).",
        "Executed: HbA1c trend shows improvement from 8.2% (Jan 2023) to 7.1% (Jun 2024). Active diagnoses: T2DM E11.65, HTN I10, Mixed Hyperlipidemia E78.2, OSA G47.33.",
    ],
    "code": [
        "Executed: Root cause identified — NullPointerException at UserService.java:142 caused by lazy-loaded profile object. Fix: Add FetchType.EAGER or JOIN FETCH to JPQL query.",
        "Executed: SQL query performing full table scan on orders (14M rows). Missing index on `created_at`. Recommended fix: CREATE INDEX idx_orders_created_at ON orders(created_at DESC).",
    ],
    "research": [
        "Executed: Smith et al. (2022) — scaling plateaus at 100B params on standard benchmarks. Chen et al. (2023) — sparse attention enables efficient scaling to 500B+. Key conflict: data quality vs. architecture as bottleneck.",
        "Executed: Ablation results — cross-layer attention: -3.4pp, rotary embeddings: -1.8pp, sparse MoE routing: -2.1pp. Cross-layer attention is the single largest contributor.",
    ],
    "customer_support": [
        "Executed: Stripe shows two charge_ids for March 2024 billing cycle — both $4,500. Duplicate confirmed. Refund initiated for charge_id ch_2nd_march via Admin > Billing > Refunds.",
        "Executed: IdP configuration shows old domain still registered. New domain added and DNS TXT record verification initiated. User instructed to re-accept invitation email.",
    ],
    "enterprise_docs": [
        "Executed: Kafka consumer group `payments-processor-v2` has lag of 1,243. Below 100k threshold — safe to restart. Command: `kubectl rollout restart deployment/payments-consumer -n production`.",
        "Executed: Incident declared P0 at 14:32 UTC. On-call paged via PagerDuty. Engineering Director notified at 14:38 UTC. #incidents-p0 thread opened.",
    ],
}

# Validator outputs — brief, conclusive
VALIDATOR_OUTPUTS = {
    "legal": "Validated: Findings are accurate and consistent with contract text. Liability cap confirmed at 12-month fee amount. Summary approved for client delivery.",
    "finance": "Validated: Revenue, margin, and segment figures cross-checked against SEC 10-Q filing. Guidance range confirmed. Summary is accurate and complete.",
    "medical": "Validated: Medication list accurate per EHR. Allergy flags cross-checked — no contraindication with proposed new medication. Clinical note approved for attending review.",
    "code": "Validated: Root cause analysis is correct. The proposed fix (FetchType.EAGER) resolves the NPE. No side effects on existing query patterns. Approved for implementation.",
    "research": "Validated: Paper findings accurately represented. Conflict between Smith et al. and Chen et al. correctly identified. Literature review section approved.",
    "customer_support": "Validated: Refund initiated correctly. Customer communication drafted and reviewed for tone and accuracy. Ticket can be marked as resolved pending customer confirmation.",
    "enterprise_docs": "Validated: Restart safe — lag below critical threshold. Steps followed per runbook ADR-091. Post-restart monitoring window started. Incident ticket updated.",
}


class HealthyMultiAgentGenerator(BaseGenerator):
    """
    Generates healthy multi-agent pipeline samples with clean handoffs,
    no repeated context, and compressed inter-agent summaries.
    Pattern: Planner → Executor → Validator
    """

    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "healthy_architectures"
    SUBCATEGORY = "healthy_multi_agent"
    GENERATOR_NAME = "healthy_multi_agent_gen_v1"

    def generate(
        self,
        count: int,
        domain: str = "all",
        difficulty: str = "medium",
        **kwargs,
    ) -> List[ContextBenchSample]:
        domains = list(SYSTEM_PROMPTS.keys()) if domain == "all" else [domain]
        samples: List[ContextBenchSample] = []

        for i in range(count):
            current_domain = domains[i % len(domains)]
            sample = self._build_sample(index=i, domain=current_domain, difficulty=difficulty)
            samples.append(sample)

        return samples

    def _build_sample(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Randomly pick which agent role is the current active agent in this context snapshot
        active_role = random.choice(["executor", "validator"])

        if active_role == "executor":
            # Snapshot: Executor receives the planner's plan + 2-3 focused chunks
            system_prompt = AGENT_ROLES["executor"]
            planner_output = random.choice(PLANNER_OUTPUTS[domain])
            available_chunks = RETRIEVAL_CHUNKS[domain]
            num_chunks = random.randint(2, 3)
            chosen_chunks = random.sample(available_chunks, min(num_chunks, len(available_chunks)))

            messages = [
                {"role": "system", "content": f"[PLANNER OUTPUT - COMPRESSED]\n{planner_output}"},
                {"role": "user", "content": random.choice(QUERIES[domain])},
            ]
            memory = []  # Healthy: executor does not carry full conversation history

        else:
            # Snapshot: Validator receives executor's results — compressed, no re-dumping
            system_prompt = AGENT_ROLES["validator"]
            executor_output = random.choice(EXECUTOR_OUTPUTS[domain])
            original_query = random.choice(QUERIES[domain])
            chosen_chunks = []  # Validator does not re-retrieve; it reviews results

            messages = [
                {"role": "user", "content": f"Original Task: {original_query}"},
                {"role": "assistant", "content": f"[EXECUTOR RESULT - COMPRESSED]\n{executor_output}"},
                {"role": "user", "content": "Please validate the above result for correctness and completeness."},
            ]
            # Optionally a single memory entry for the validator
            memory = (
                [random.choice(MEMORY_SNIPPETS[domain])] if random.random() > 0.5 else []
            )

        context = ContextData(
            system_prompt=system_prompt,
            messages=messages,
            retrieval_chunks=chosen_chunks,
            tool_outputs=[],  # Healthy agents don't dump raw tool outputs
            memory=memory,
        )

        total_text = (
            system_prompt
            + " ".join(m["content"] for m in messages)
            + " ".join(chosen_chunks)
            + " ".join(memory)
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
                "active_role": active_role,
                "num_retrieval_chunks": len(chosen_chunks),
                "memory_entries": len(memory),
                "pattern": "planner_executor_validator",
            },
        )

        metadata = BenchmarkMetadata(
            token_count=approx_tokens,
            version=self.version,
        )

        return ContextBenchSample(
            id=f"cb_v1_healthy_agent_{index:04d}_{domain}_{active_role}",
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
