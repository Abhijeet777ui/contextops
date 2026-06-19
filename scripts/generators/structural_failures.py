"""
StructuralFailureGenerator — Phase 3: Architecture Quality Failures

Three failure subtypes:
  1. system_prompt_bloat  — system prompt dominates the context window
  2. retrieval_flooding   — far too many chunks retrieved (top-K discipline broken)
  3. memory_explosion     — stale conversation history dominates the window

Each sample has:
  - A realistic domain-grounded context
  - ground_truth.architecture_failure = True
  - ground_truth.expected_properties that accurately describe the failure
  - ground_truth.failure_modes listing the detected issues
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
)

# ---------------------------------------------------------------------------
# Bloat patterns — realistic additions that make system prompts balloon
# ---------------------------------------------------------------------------
BLOAT_ADDITIONS = [
    "\n\nFormatting Rules:\n- Always respond in Markdown.\n- Use headers (##) for each main section.\n- Use bullet points for lists.\n- Use bold for key terms.\n- Use tables when comparing more than 2 items.\n- Never use passive voice.\n- Keep sentences under 20 words.\n- Use active verbs.\n- Avoid jargon unless defined first.\n- Always include a TL;DR at the top.",

    "\n\nTone and Style Guidelines:\n- Be professional but approachable.\n- Avoid humor unless the user explicitly requests it.\n- Do not use contractions in formal contexts.\n- Mirror the user's communication style.\n- Prefer concise answers but expand when asked.\n- Always acknowledge the user's effort before providing critique.\n- Never start a response with 'I'.\n- Do not repeat the user's question back to them verbatim.",

    "\n\nCompliance and Legal Disclaimers:\n- This assistant does not provide legal, medical, financial, or tax advice.\n- All information is provided for informational purposes only.\n- Users should consult qualified professionals for specific advice.\n- The organization is not liable for decisions made based on this assistant's outputs.\n- Confidential information should not be entered into this system.\n- All outputs may be reviewed for quality assurance purposes.\n- Outputs do not represent the official position of the organization.",

    "\n\nSafety and Content Policy:\n- Never generate content that promotes violence, hate speech, or discrimination.\n- Do not assist with illegal activities.\n- Refuse requests for personal information about real individuals.\n- Do not generate explicit or adult content.\n- If a request seems harmful, ask for clarification before proceeding.\n- Always prioritize user safety above task completion.\n- Escalate to a human agent if the user appears to be in distress.",

    "\n\nResponse Structure Requirements:\n- Begin every response with a one-sentence summary.\n- Follow with the detailed answer.\n- End with 'Next Steps' or 'Recommended Actions' where applicable.\n- Always cite your sources when drawing from provided context.\n- If you are uncertain, say so explicitly rather than guessing.\n- Provide confidence levels for factual claims where relevant.\n- Limit responses to 500 words unless the user requests more detail.",

    "\n\nPersonalization Instructions:\n- Address the user by their first name if known.\n- Remember the user's stated preferences throughout the session.\n- Adapt your technical depth to the user's apparent expertise level.\n- If the user has made a previous request in this session, reference it for continuity.\n- Do not repeat information already provided unless asked.\n- Proactively suggest related topics the user might find useful.\n- Track the user's goals and align responses accordingly.",

    "\n\nKnowledge Boundaries:\n- Your training data has a cutoff date. Acknowledge this when discussing recent events.\n- Do not speculate about information not present in the provided context.\n- Distinguish clearly between facts from the context and general knowledge.\n- If the context is insufficient, say so rather than hallucinating.\n- Prioritize information from the provided retrieval chunks over general knowledge.\n- When multiple sources conflict, flag the discrepancy explicitly.",
]

# ---------------------------------------------------------------------------
# Stale memory turns — realistic but outdated conversation entries
# ---------------------------------------------------------------------------
STALE_MEMORY_TURNS = {
    "legal": [
        "[2024-01-15] User asked about NDA clauses for a previous vendor agreement (now closed).",
        "[2024-01-22] User requested summary of force majeure provisions — delivered.",
        "[2024-02-03] User asked about governing law for a UK contract — resolved.",
        "[2024-02-18] User wanted to understand arbitration clause for Contract B (expired).",
        "[2024-03-01] User inquired about IP assignment terms — previous project, now archived.",
        "[2024-03-12] User asked about breach of contract remedies for Vendor X dispute — settled.",
        "[2024-04-05] User requested a summary of the employment agreement with John Doe — terminated.",
        "[2024-04-20] User asked about GDPR data processing clauses for EU vendor — completed.",
        "[2024-05-09] User asked about M&A reps and warranties for Deal Y — deal closed.",
        "[2024-05-23] User wanted indemnification analysis for old SLA — superseded by new contract.",
        "[2024-06-01] User asked about limitation of liability caps — previous session.",
        "[2024-06-08] User requested Delaware case law on fiduciary duty — delivered last week.",
    ],
    "finance": [
        "[2024-01-10] User requested Q1 FY2024 earnings summary for MSFT — delivered.",
        "[2024-01-25] User asked about Apple's Services revenue trend Q4 FY2023 — old data.",
        "[2024-02-14] User wanted breakdown of Google's ad revenue — previous quarter.",
        "[2024-02-28] User requested comparison of FAANG free cash flows for FY2023.",
        "[2024-03-15] User asked about Meta's Reality Labs losses — delivered.",
        "[2024-04-02] User requested Amazon AWS revenue segment breakdown — Q1 data now stale.",
        "[2024-04-18] User wanted guidance range for NVDA Q2 FY2024 — passed.",
        "[2024-05-07] User asked about TSLA gross margin compression — previous quarter.",
        "[2024-05-22] User requested S&P 500 sector performance summary for March 2024.",
        "[2024-06-03] User asked about Fed rate decision impact on tech sector — old event.",
        "[2024-06-10] User requested Apple Q2 FY2024 summary — now superseded by Q3 data.",
        "[2024-06-15] User asked about Alphabet's YouTube ad revenue — prior session.",
    ],
    "medical": [
        "[2024-01-08] Patient P-20834 medication review completed — Metformin dose adjusted.",
        "[2024-01-20] Lab results reviewed: HbA1c 8.2% — action taken.",
        "[2024-02-05] Patient asked about Lisinopril side effects — counseled.",
        "[2024-02-19] Reviewed MRI report from 2022 — findings documented.",
        "[2024-03-10] Patient reported mild dizziness — attributed to Lisinopril, monitored.",
        "[2024-03-25] Discussed sleep apnea management options — CPAP referral made.",
        "[2024-04-08] HbA1c follow-up: 7.8% — improvement noted.",
        "[2024-04-22] Patient asked about statin alternatives — Atorvastatin continued.",
        "[2024-05-06] Reviewed overdue colonoscopy screening — patient agreed to schedule.",
        "[2024-05-20] Patient reported improved sleep quality with CPAP.",
        "[2024-06-03] HbA1c 7.4% — continued improvement.",
        "[2024-06-10] Current visit: HbA1c 7.1% — medication regimen unchanged.",
    ],
    "code": [
        "[2024-01-12] Fixed memory leak in image processing pipeline — PR #1204 merged.",
        "[2024-01-28] Reviewed auth service JWT validation logic — no issues found.",
        "[2024-02-09] Debugged slow database queries in reporting service — index added.",
        "[2024-02-24] Code review for payment service refactor — approved with minor comments.",
        "[2024-03-07] Fixed race condition in notification worker — PR #1389 merged.",
        "[2024-03-19] Reviewed API rate limiting implementation — approved.",
        "[2024-04-02] Debugged Kafka consumer lag issue — consumer group rebalanced.",
        "[2024-04-16] Fixed broken CI pipeline for staging environment — resolved.",
        "[2024-05-01] Reviewed microservice migration plan — architecture approved.",
        "[2024-05-15] Debugged 403 error in admin panel — permissions config corrected.",
        "[2024-05-29] Fixed CSV memory issue in data export service — chunked reading added.",
        "[2024-06-12] Reviewed DDD refactoring proposal — approved for Q3 implementation.",
    ],
    "research": [
        "[2024-01-14] Summarized Hendrycks et al. (2021) MMLU paper — delivered.",
        "[2024-01-29] Reviewed BIG-Bench methodology — notes added to literature review.",
        "[2024-02-11] Summarized Kaplan et al. (2020) scaling laws paper.",
        "[2024-02-26] Compared GPT-4 and Claude evaluation benchmarks.",
        "[2024-03-09] Reviewed Hoffmann et al. (2022) Chinchilla paper.",
        "[2024-03-22] Summarized Smith et al. (2022) scaling plateau findings.",
        "[2024-04-05] Reviewed Chen et al. (2023) sparse attention paper.",
        "[2024-04-19] Drafted literature review section on evaluation methodology.",
        "[2024-05-03] Reviewed contamination-free evaluation methodology paper.",
        "[2024-05-17] Summarized ablation study results from target paper.",
        "[2024-05-31] Compared MMLU, HumanEval, and GSM8K benchmark designs.",
        "[2024-06-14] Finalized comparison table for Smith and Chen papers.",
    ],
    "customer_support": [
        "[2024-01-11] Customer reported slow dashboard load times — resolved by engineering.",
        "[2024-01-26] Customer requested bulk user import feature — forwarded to product.",
        "[2024-02-08] Customer had billing discrepancy — credit of $120 applied.",
        "[2024-02-23] Customer asked about SAML SSO setup — documentation shared.",
        "[2024-03-07] Customer requested data export for compliance audit — provided.",
        "[2024-03-21] Customer reported API rate limiting issues — limits adjusted.",
        "[2024-04-04] Customer asked about SLA uptime commitments — 99.9% confirmed.",
        "[2024-04-18] Customer requested custom webhook for Salesforce — configured.",
        "[2024-05-02] Customer added 50 new seats to Business plan.",
        "[2024-05-16] Customer reported duplicate charge in March — under investigation.",
        "[2024-05-30] Customer asked about EU data residency — confirmed eu-west-1.",
        "[2024-06-13] Customer escalated dashboard issue for EU workspace — engineering engaged.",
    ],
    "enterprise_docs": [
        "[2024-01-09] Engineer provisioned staging DB for Payments team.",
        "[2024-01-24] ADR-045 approved: migration to Kafka for event streaming.",
        "[2024-02-06] On-call rotation updated for Q1 2024.",
        "[2024-02-20] Reviewed service catalog onboarding for Analytics service.",
        "[2024-03-05] P0 incident: auth service outage — resolved in 47 minutes.",
        "[2024-03-18] ADR-047 approved: OAuth 2.0 migration for auth service.",
        "[2024-04-01] Code freeze policy updated for Q2 release cycle.",
        "[2024-04-15] New pre-approved tool added: Notion.",
        "[2024-04-29] Kafka consumer group restarted for payments-processor-v1.",
        "[2024-05-13] Staging DB for ML team provisioned.",
        "[2024-05-27] P1 incident: EU dashboard outage — resolved in 2.5 hours.",
        "[2024-06-11] ADR-051 submitted: data residency expansion to APAC.",
    ],
}


class StructuralFailureGenerator(BaseGenerator):
    """
    Generates structurally broken context samples.
    Subtypes: system_prompt_bloat, retrieval_flooding, memory_explosion.
    """

    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "structural_failures"
    GENERATOR_NAME = "structural_failure_gen_v1"

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
            ["system_prompt_bloat", "retrieval_flooding", "memory_explosion"]
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

    # ------------------------------------------------------------------
    def _build_sample(
        self, index: int, domain: str, difficulty: str, subtype: str
    ) -> ContextBenchSample:
        if subtype == "system_prompt_bloat":
            return self._system_prompt_bloat(index, domain, difficulty)
        elif subtype == "retrieval_flooding":
            return self._retrieval_flooding(index, domain, difficulty)
        else:
            return self._memory_explosion(index, domain, difficulty)

    # ------------------------------------------------------------------
    def _system_prompt_bloat(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        base_prompt = SYSTEM_PROMPTS[domain]
        # Add 3-5 bloat blocks to the system prompt
        num_bloat = {"easy": 2, "medium": 4, "hard": 6}.get(difficulty, 4)
        additions = random.sample(BLOAT_ADDITIONS, min(num_bloat, len(BLOAT_ADDITIONS)))
        bloated_prompt = base_prompt + "".join(additions)

        query = random.choice(QUERIES[domain])
        available_chunks = RETRIEVAL_CHUNKS[domain]
        chunks = random.sample(available_chunks, min(3, len(available_chunks)))

        context = ContextData(
            system_prompt=bloated_prompt,
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=[],
            memory=[],
        )
        total_text = bloated_prompt + query + " ".join(chunks)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["system_prompt_bloat"],
            severity=_severity_by_tokens(approx_tokens),
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=False,
                contains_density_bloat=True,
                contains_structure_imbalance=True,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "system_prompt_bloat", "bloat_blocks_added": num_bloat},
        )
        return ContextBenchSample(
            id=f"cb_v1_struct_bloat_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="system_prompt_bloat",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    # ------------------------------------------------------------------
    def _retrieval_flooding(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Healthy = 3-5 chunks. Flooding = 15-30 chunks via repetition with variation.
        num_flood = {"easy": 12, "medium": 18, "hard": 28}.get(difficulty, 18)
        available = RETRIEVAL_CHUNKS[domain]
        # Cycle through available chunks repeatedly to simulate flooding
        flooded_chunks = [available[i % len(available)] for i in range(num_flood)]

        query = random.choice(QUERIES[domain])
        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=flooded_chunks,
            tool_outputs=[],
            memory=[],
        )
        total_text = SYSTEM_PROMPTS[domain] + query + " ".join(flooded_chunks)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["retrieval_flooding", "redundancy_via_cycling"],
            severity=_severity_by_tokens(approx_tokens),
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=True,
                contains_density_bloat=True,
                contains_structure_imbalance=True,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "retrieval_flooding", "num_chunks_injected": num_flood},
        )
        return ContextBenchSample(
            id=f"cb_v1_struct_flood_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="retrieval_flooding",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    # ------------------------------------------------------------------
    def _memory_explosion(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Healthy = 0-1 memory entries. Explosion = 8-12 stale turns.
        num_turns = {"easy": 6, "medium": 10, "hard": 12}.get(difficulty, 10)
        stale_turns = STALE_MEMORY_TURNS.get(domain, [])
        # Cycle if needed
        memory = [stale_turns[i % len(stale_turns)] for i in range(num_turns)]

        query = random.choice(QUERIES[domain])
        chunks = random.sample(RETRIEVAL_CHUNKS[domain], min(3, len(RETRIEVAL_CHUNKS[domain])))

        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=[],
            memory=memory,
        )
        total_text = SYSTEM_PROMPTS[domain] + query + " ".join(chunks) + " ".join(memory)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["memory_explosion", "stale_context_injection"],
            severity=_severity_by_tokens(approx_tokens),
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=True,
                contains_density_bloat=True,
                contains_structure_imbalance=True,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={"subtype": "memory_explosion", "stale_memory_turns": num_turns},
        )
        return ContextBenchSample(
            id=f"cb_v1_struct_mem_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="memory_explosion",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )


def _severity_by_tokens(tokens: int) -> str:
    if tokens < 500:
        return "low"
    elif tokens < 1500:
        return "medium"
    elif tokens < 4000:
        return "high"
    else:
        return "critical"
