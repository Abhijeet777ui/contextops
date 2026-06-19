"""
RedundancyFailureGenerator — Phase 3: Redundancy Failures

Two failure subtypes:
  1. near_duplicate_cluster  — multiple chunks with very similar meaning (paraphrased)
  2. boilerplate_explosion   — repeated headers, footers, legal disclaimers from web scraping

These are insidious failures because the content looks "valid" — it's not noise,
it's just needlessly repeated information that wastes the context window.
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
# Paraphrase clusters — semantically identical content, different wording.
# These simulate what happens when a poorly configured retriever pulls from
# multiple sources that all describe the same fact.
# ---------------------------------------------------------------------------
PARAPHRASE_CLUSTERS = {
    "legal": [
        # All say: "liability is capped at 12 months of fees"
        "The maximum aggregate liability of either party shall not exceed the total fees paid in the twelve months preceding the claim.",
        "Neither party's total liability will surpass the amount paid by the customer during the one-year period before the incident occurred.",
        "Aggregate liability is capped at fees received over the prior 12-month period from the date the claim arises.",
        "The liability cap is set to the sum of all fees paid in the twelve months leading up to the event giving rise to liability.",
        "In no case shall cumulative liability exceed what the customer paid over the year prior to the disputed event.",
        "Total liability exposure for both parties is limited to 12 months of paid fees preceding the relevant claim date.",
        "The parties agree that liability shall not exceed fees remitted in the twelve-month window before the claim.",
        "Maximum recoverable damages are bounded by the fees paid within one year before the triggering event.",
    ],
    "finance": [
        # All say: "Apple Q3 FY2024 revenue was $85.8B, up 5%"
        "Apple reported third quarter fiscal 2024 revenue of $85.8 billion, a 5% increase compared to the year-ago period.",
        "In Q3 FY2024, Apple's total revenue reached $85.8B, growing 5% year-over-year from Q3 FY2023.",
        "Apple's Q3 results showed revenue of 85.8 billion dollars, up 5 percent on an annual basis.",
        "The Cupertino-based company posted $85.8B in revenue for Q3 FY2024, beating the prior year's figure by 5%.",
        "Apple generated $85.8 billion in revenue during its fiscal third quarter of 2024, up 5% from the same period last year.",
        "Q3 FY2024 revenue for Apple came in at $85.8 billion — a year-on-year improvement of approximately 5%.",
        "Apple's top-line for the July 2024 quarter was $85.8B, representing 5% growth versus Q3 FY2023.",
        "Revenue of $85.8 billion was reported by Apple in Q3 FY2024, reflecting a 5% increase year-over-year.",
    ],
    "medical": [
        # All say: "Patient's HbA1c improved to 7.1%"
        "The patient's most recent HbA1c measurement was 7.1%, reflecting improved glycemic control.",
        "HbA1c has improved to 7.1% as of the June 2024 visit, down from 8.2% in January 2023.",
        "Latest lab results show an HbA1c of 7.1%, indicating good progress in diabetes management.",
        "Glycated hemoglobin (HbA1c) is currently at 7.1%, showing steady improvement over 18 months.",
        "The patient's blood sugar control has improved, with HbA1c at 7.1% in the most recent test.",
        "A June 2024 HbA1c result of 7.1% demonstrates continued response to the current treatment regimen.",
        "HbA1c trend: patient has achieved 7.1% as of June 2024, down from a baseline of 8.2%.",
        "Current HbA1c: 7.1% — an improvement of 1.1 percentage points from the 2023 baseline.",
    ],
    "code": [
        # All say: "NullPointerException caused by lazy-loaded profile at UserService.java:142"
        "The NullPointerException at UserService.java:142 is caused by accessing a null profile object that was not loaded.",
        "Root cause: `user.getProfile()` returns null because the profile association is lazily loaded and not initialized.",
        "Line 142 of UserService.java throws NPE because the JPA entity's profile field is not eagerly fetched.",
        "The stack trace points to UserService.java:142 where a null profile is dereferenced — a lazy-loading issue.",
        "Profile object is null at UserService line 142 due to JPA lazy fetch strategy not loading the association.",
        "NPE originates from uninitialized profile in UserService:142; the ORM is not pre-loading this relationship.",
        "The user profile relationship is fetched lazily, resulting in a null reference exception at line 142 of UserService.",
        "Error at UserService.java:142: profile is null because FetchType.LAZY prevents automatic loading of the entity.",
    ],
    "research": [
        # All say: "Smith et al. found scaling plateaus at 100B parameters"
        "Smith et al. (2022) demonstrated that scaling transformer models beyond 100 billion parameters yields diminishing returns on standard benchmarks.",
        "According to Smith et al. 2022, performance improvements plateau when model size exceeds 100B parameters.",
        "The Smith et al. (2022) paper shows a scaling plateau effect above 100 billion parameters on benchmark evaluations.",
        "Smith and colleagues (2022) found that beyond 100B parameters, benchmark perplexity improvements diminish significantly.",
        "Scaling laws from Smith et al. (2022) suggest that 100B parameters marks a threshold of diminishing gains on standard NLP benchmarks.",
        "Research by Smith et al. (2022) indicates that scaling past 100 billion parameters no longer yields proportional performance improvements.",
        "Smith et al. 2022 showed that the return on adding parameters drops sharply once models exceed 100B in size.",
        "The 2022 study by Smith and collaborators established that benchmark gains plateau around the 100B parameter mark.",
    ],
    "customer_support": [
        # All say: "The duplicate charge will be refunded in 5-7 business days"
        "The duplicate charge has been identified and a refund will be processed within 5 to 7 business days.",
        "We have initiated a refund for the duplicate payment; customers typically see this resolved in 5-7 business days.",
        "A refund for the duplicate March charge is underway and should appear in 5-7 business days.",
        "The customer's duplicate charge is being reversed; the refund window is 5 to 7 business days.",
        "Refund processing has begun for the duplicate billing; expected timeframe is five to seven business days.",
        "The duplicate charge will be refunded in the next 5-7 business days per standard billing procedures.",
        "Processing time for the refund on the duplicate transaction is 5-7 business days from the initiation date.",
        "The customer should receive the refund within 5-7 business days once the duplicate is confirmed and reversed.",
    ],
    "enterprise_docs": [
        # All say: "Kafka consumer lag above 100k should page on-call"
        "If the Kafka consumer group lag exceeds 100,000 messages, the on-call engineer must be paged immediately.",
        "Consumer lag above 100k is the threshold for triggering a PagerDuty alert to the on-call engineer.",
        "When Kafka lag surpasses 100,000, alert the on-call rotation via PagerDuty without delay.",
        "A consumer group lag of 100k or more constitutes a warning condition requiring immediate on-call notification.",
        "The on-call runbook specifies: page on-call if consumer lag exceeds 100,000 messages.",
        "Kafka monitoring policy: lag > 100k messages triggers an immediate PagerDuty page to the on-call engineer.",
        "100,000 messages of consumer lag is the defined alerting threshold for engaging the on-call team.",
        "Per the incident response policy, consumer group lag exceeding 100k requires an on-call page via PagerDuty.",
    ],
}

# ---------------------------------------------------------------------------
# Boilerplate patterns — repeated structural noise from web scraping / doc systems
# ---------------------------------------------------------------------------
BOILERPLATE_BLOCKS = [
    "--- Page 1 of 12 --- Document ID: DOC-2024-REV3 --- Confidential --- Internal Use Only ---",
    "This document is proprietary and confidential. Unauthorized reproduction or distribution is prohibited. All rights reserved.",
    "DISCLAIMER: The information contained in this document is for informational purposes only and does not constitute legal, financial, or professional advice.",
    "--- End of Section --- Continue to Next Section --- Table of Contents on Page 2 ---",
    "Copyright 2024 Acme Corporation. All rights reserved. Printed copies are uncontrolled. Always refer to the online version for the latest revision.",
    "For internal distribution only. Do not share outside the organization without prior written approval from the Legal department.",
    "This page intentionally left blank.",
    "--- Document Version 3.2 --- Last Modified: 2024-06-01 --- Approved By: John Smith --- Status: FINAL ---",
    "NOTICE: This communication may contain privileged and/or confidential information. If you are not the intended recipient, please notify the sender immediately.",
    "Revision History: v1.0 (2023-01-15) Initial release | v2.0 (2023-06-01) Updated policy | v3.0 (2024-01-10) Annual review | v3.2 (2024-06-01) Minor edits.",
]


class RedundancyFailureGenerator(BaseGenerator):
    """
    Generates context samples with redundancy failures.
    Subtypes: near_duplicate_cluster, boilerplate_explosion.
    """

    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "redundancy_failures"
    GENERATOR_NAME = "redundancy_failure_gen_v1"

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
            ["near_duplicate_cluster", "boilerplate_explosion"]
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
        if subtype == "near_duplicate_cluster":
            return self._near_duplicate_cluster(index, domain, difficulty)
        else:
            return self._boilerplate_explosion(index, domain, difficulty)

    def _near_duplicate_cluster(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # How many paraphrases to inject (on top of 1-2 legitimate chunks)
        num_dupes = {"easy": 4, "medium": 6, "hard": 8}.get(difficulty, 6)

        cluster = PARAPHRASE_CLUSTERS.get(domain, [])
        paraphrases = cluster[:num_dupes]

        # Add 1-2 genuinely different chunks to make it realistic
        available = RETRIEVAL_CHUNKS[domain]
        legit_chunks = random.sample(available, min(2, len(available)))
        all_chunks = paraphrases + legit_chunks
        random.shuffle(all_chunks)

        query = random.choice(QUERIES[domain])
        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=all_chunks,
            tool_outputs=[],
            memory=[],
        )
        total_text = SYSTEM_PROMPTS[domain] + query + " ".join(all_chunks)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["near_duplicate_cluster"],
            severity="high" if num_dupes >= 6 else "medium",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=True,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=True,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={
                "subtype": "near_duplicate_cluster",
                "paraphrase_count": num_dupes,
                "legit_chunks": len(legit_chunks),
            },
        )
        return ContextBenchSample(
            id=f"cb_v1_redun_dupcluster_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="near_duplicate_cluster",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    def _boilerplate_explosion(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        # Inject repeated boilerplate between real chunks
        num_boilerplate = {"easy": 3, "medium": 5, "hard": 8}.get(difficulty, 5)
        available = RETRIEVAL_CHUNKS[domain]
        legit_chunks = random.sample(available, min(2, len(available)))

        # Cycle boilerplate blocks
        boilerplate = [BOILERPLATE_BLOCKS[i % len(BOILERPLATE_BLOCKS)] for i in range(num_boilerplate)]

        # Interleave boilerplate with real content
        all_chunks = []
        for i, chunk in enumerate(legit_chunks):
            all_chunks.append(chunk)
            if i < len(boilerplate):
                all_chunks.append(boilerplate[i])
        all_chunks.extend(boilerplate[len(legit_chunks):])

        query = random.choice(QUERIES[domain])
        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=all_chunks,
            tool_outputs=[],
            memory=[],
        )
        total_text = SYSTEM_PROMPTS[domain] + query + " ".join(all_chunks)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["boilerplate_explosion"],
            severity="medium" if num_boilerplate <= 5 else "high",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=True,
                contains_density_bloat=True,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={
                "subtype": "boilerplate_explosion",
                "boilerplate_blocks_injected": num_boilerplate,
                "legit_chunks": len(legit_chunks),
            },
        )
        return ContextBenchSample(
            id=f"cb_v1_redun_boilerplate_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="boilerplate_explosion",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )
