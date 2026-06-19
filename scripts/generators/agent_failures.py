"""
AgentFailureGenerator — Phase 3: Agent Architecture Failures

Three failure subtypes:
  1. tool_chain_bloat        — raw unpaginated API/DB output dumped directly into context
  2. recursive_planning      — agent repeats planning steps in messages, creating loop bloat
  3. multi_agent_explosion   — each agent dumps its full raw output to the next, compounding

These failures are unique to agentic pipelines and represent a rapidly growing
class of real-world context architecture problems as multi-agent systems scale.
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
from .domain_templates import SYSTEM_PROMPTS, QUERIES, RETRIEVAL_CHUNKS, AGENT_ROLES

# ---------------------------------------------------------------------------
# Raw, unpaginated tool outputs — realistic but massively bloated
# ---------------------------------------------------------------------------
RAW_TOOL_OUTPUTS = {
    "legal": (
        '{"search_results": ['
        + ", ".join([
            f'{{"id": "doc_{i}", "source": "contract_archive_{i}.pdf", "page": {random.randint(1,50)}, "content": "Section {i}.{random.randint(1,9)}: The party of the first part agrees to indemnify, defend, and hold harmless the party of the second part from any and all claims, damages, losses, costs, and expenses, including reasonable attorneys fees, arising out of or resulting from any breach of this agreement, negligent or wrongful act or omission, or violation of applicable law by the indemnifying party, its officers, directors, employees, agents, or subcontractors.", "relevance_score": {round(random.uniform(0.1, 0.99), 2)}}}'
            for i in range(40)
        ])
        + '], "total_docs_searched": 12400, "query_time_ms": 824}'
    ),
    "finance": (
        '{"financial_data": {"ticker": "AAPL", "period": "Q3_FY2024", "raw_rows": ['
        + ", ".join([
            f'{{"quarter": "Q{random.randint(1,4)}_FY{random.randint(2020,2024)}", "revenue": {random.randint(70,90)}800000000, "gross_margin": {round(random.uniform(38,48), 1)}, "operating_income": {random.randint(20,30)}000000000, "net_income": {random.randint(18,25)}000000000, "eps_diluted": {round(random.uniform(1.2, 1.6), 2)}, "shares_outstanding": {random.randint(15,16)}500000000, "cash_equivalents": {random.randint(50,70)}000000000, "long_term_debt": {random.randint(90,110)}000000000}}'
            for i in range(50)
        ])
        + '], "metadata": {"source": "SEC_EDGAR", "rows_returned": 50, "total_available": 84}}}'
    ),
    "medical": (
        '{"patient_records": {"patient_id": "P-20834", "all_encounters": ['
        + ", ".join([
            f'{{"encounter_id": "ENC-{10000+i}", "date": "202{random.randint(2,4)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "type": "{random.choice(["office_visit","lab","imaging","phone"])}", "provider": "Dr. {random.choice(["Chen","Park","Williams","Johnson"])}", "notes": "Patient presented with {random.choice(["routine follow-up","medication review","acute complaint","preventive care"])}. Vitals stable. Plan updated. Follow-up in {random.randint(1,12)} weeks.", "icd10_codes": ["{random.choice(["E11.65","I10","E78.2","G47.33","Z00.00"])}"], "medications_reviewed": ["Metformin 1000mg BID", "Lisinopril 10mg QD"]}}'
            for i in range(45)
        ])
        + '], "total_encounters": 45}}'
    ),
    "code": (
        '{"search_results": {"query": "NullPointerException", "repo": "monorepo", "matches": ['
        + ", ".join([
            f'{{"file": "src/{random.choice(["UserService","OrderService","PaymentService","NotificationService","AuthService"])}.java", "line": {random.randint(100,500)}, "column": {random.randint(1,50)}, "snippet": "{random.choice(["user.getProfile().getName()", "order.getCustomer().getEmail()", "payment.getMethod().getToken()", "notification.getRecipient().getId()", "auth.getSession().getUserId()"])}", "severity": "{random.choice(["HIGH","MEDIUM","LOW"])}", "last_modified": "202{random.randint(3,4)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "author": "{random.choice(["alice","bob","charlie","diana","eve"])}", "commit": "{hex(random.randint(100000,999999))[2:]}abc"}}'
            for i in range(60)
        ])
        + '], "total_matches": 60, "scan_time_ms": 1240}}'
    ),
    "research": (
        '{"papers": ['
        + ", ".join([
            f'{{"id": "paper_{i}", "title": "Advances in {random.choice(["Transformer","Attention","Scaling","Alignment","RLHF","RAG","Fine-tuning"])} Research ({2020+i%5})", "authors": ["{random.choice(["Smith","Chen","Johnson","Park","Williams","Garcia"])} et al."], "year": {2020+i%5}, "venue": "{random.choice(["NeurIPS","ICML","ICLR","ACL","EMNLP"])}", "citations": {random.randint(50,5000)}, "abstract": "We present a novel approach to {random.choice(["scaling","alignment","evaluation","fine-tuning"])} that demonstrates significant improvements over prior state-of-the-art methods on standard benchmarks including MMLU, HumanEval, and GSM8K.", "doi": "10.{random.randint(1000,9999)}/arxiv.{2020+i%5}.{random.randint(10000,99999)}", "pdf_url": "https://arxiv.org/pdf/{2020+i%5}.{random.randint(10000,99999)}.pdf"}}'
            for i in range(50)
        ])
        + '], "total_results": 50, "query_time_ms": 312}}'
    ),
    "customer_support": (
        '{"customer_history": {"customer_id": "CUS-88821", "all_tickets": ['
        + ", ".join([
            f'{{"ticket_id": "TKT-{20000+i}", "created_at": "202{random.randint(2,4)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "status": "{random.choice(["resolved","closed","open","escalated"])}", "category": "{random.choice(["billing","access","feature_request","bug","integration"])}", "priority": "{random.choice(["P1","P2","P3","P4"])}", "subject": "{random.choice(["Login issue","Billing discrepancy","API error","Dashboard not loading","Webhook configuration","Seat management","Data export"])}", "resolution_time_hours": {random.randint(1,72)}, "csat_score": {random.randint(1,5)}, "assigned_agent": "{random.choice(["Agent_A","Agent_B","Agent_C","Agent_D"])}"}}'
            for i in range(55)
        ])
        + '], "total_tickets": 55}}'
    ),
    "enterprise_docs": (
        '{"audit_log": {"service": "payments-consumer", "events": ['
        + ", ".join([
            f'{{"event_id": "EVT-{50000+i}", "timestamp": "2024-{random.randint(1,6):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}Z", "event_type": "{random.choice(["consumer_restart","lag_alert","rebalance","offset_commit","partition_reassign","consumer_join","consumer_leave"])}", "consumer_group": "payments-processor-v2", "partition": {random.randint(0,11)}, "offset": {random.randint(100000,999999)}, "lag": {random.randint(0,50000)}, "actor": "{random.choice(["k8s_operator","on_call_engineer","automated_scaler","deployment_pipeline"])}", "details": "Event processed successfully. No action required."}}'
            for i in range(60)
        ])
        + '], "total_events": 60}}'
    ),
}

# ---------------------------------------------------------------------------
# Recursive planning messages — agent that keeps re-planning the same task
# ---------------------------------------------------------------------------
RECURSIVE_PLAN_MESSAGES = {
    "legal": [
        {"role": "assistant", "content": "Plan v1: (1) Retrieve indemnification clauses. (2) Identify liability caps. (3) Summarize for client."},
        {"role": "assistant", "content": "Revised Plan v2: (1) Retrieve indemnification clauses from contract. (2) Check for cross-references to Section 14. (3) Identify liability caps. (4) Note any carve-outs. (5) Summarize for client."},
        {"role": "assistant", "content": "Updated Plan v3: (1) Retrieve all indemnification clauses including cross-references. (2) Map to liability limitation section. (3) Identify monetary caps and carve-outs. (4) Note governing law implications. (5) Draft client summary with citations."},
        {"role": "assistant", "content": "Final Plan v4: Same as v3 but I should also check if there are any indemnification carve-outs in the force majeure clause. Let me re-retrieve all relevant sections before proceeding."},
        {"role": "assistant", "content": "Plan v4 revised: I realize I should start from the beginning. (1) Full contract review. (2) Index all clauses. (3) Retrieve indemnification. (4) Retrieve liability caps. (5) Retrieve force majeure. (6) Synthesize. (7) Summarize."},
    ],
    "finance": [
        {"role": "assistant", "content": "Plan v1: Retrieve Q3 revenue, gross margin, and guidance for the summary."},
        {"role": "assistant", "content": "Revised Plan v2: Retrieve Q3 revenue, gross margin, segment breakdown, and guidance. Also check FCF."},
        {"role": "assistant", "content": "Updated Plan v3: Get Q3 revenue, gross margin, segment data, guidance, FCF, and share buyback info. Should also compare to Q3 FY2023."},
        {"role": "assistant", "content": "Plan v4: I think I need a more systematic approach. Retrieve all Q3 metrics, then all Q3 FY2023 metrics, then compute all YoY deltas, then retrieve guidance, then check buyback authorization, then check net debt."},
        {"role": "assistant", "content": "Plan v5: Actually let me restart. I'll pull the full 10-Q and parse every section systematically rather than retrieving individual metrics."},
    ],
    "medical": [
        {"role": "assistant", "content": "Plan v1: Review current medications and check for interactions."},
        {"role": "assistant", "content": "Plan v2: Review medications, check interactions, then review latest labs."},
        {"role": "assistant", "content": "Plan v3: Review all medications, check drug-drug interactions, check drug-allergy interactions, review HbA1c trend, review MRI findings, check overdue screenings."},
        {"role": "assistant", "content": "Plan v4: I should be more systematic. Start with full medication reconciliation. Then allergy cross-check. Then lab trend analysis. Then imaging review. Then preventive care gaps. Then draft clinical note."},
        {"role": "assistant", "content": "Plan v5: Wait — I need to re-check if the patient's allergy to sulfonamides affects any of the current medications first. Let me restart the medication review."},
    ],
    "code": [
        {"role": "assistant", "content": "Plan v1: Parse stack trace and identify root cause."},
        {"role": "assistant", "content": "Plan v2: Parse stack trace, identify root cause, then look at the UserService code."},
        {"role": "assistant", "content": "Plan v3: Parse stack trace, look at UserService.java:142, check JPA fetch type, identify fix, assess side effects."},
        {"role": "assistant", "content": "Plan v4: Actually I should check all places where getProfile() is called to understand the full blast radius before proposing a fix."},
        {"role": "assistant", "content": "Plan v5: Restarting analysis. Step 1: Get full stack trace. Step 2: Map call chain from controller to UserService. Step 3: Identify all lazy-loaded relationships. Step 4: Assess ORM configuration. Step 5: Propose targeted fix."},
    ],
    "research": [
        {"role": "assistant", "content": "Plan v1: Summarize methodology and findings from the papers."},
        {"role": "assistant", "content": "Plan v2: Summarize methodology, findings, and limitations from both papers. Then compare."},
        {"role": "assistant", "content": "Plan v3: Extract methodology from Smith et al. Extract methodology from Chen et al. Compare approaches. Extract findings. Compare findings. Identify conflicts. Summarize limitations. Draft synthesis."},
        {"role": "assistant", "content": "Plan v4: I think I should also look at the datasets and evaluation protocols used by each paper before comparing findings."},
        {"role": "assistant", "content": "Plan v5: Let me take a step back and re-read all papers from the beginning before attempting synthesis. The comparison is more nuanced than I initially thought."},
    ],
    "customer_support": [
        {"role": "assistant", "content": "Plan v1: Look up account, identify duplicate charge, initiate refund."},
        {"role": "assistant", "content": "Plan v2: Look up account in Stripe, identify duplicate charge_id, confirm amount, initiate refund, notify customer."},
        {"role": "assistant", "content": "Plan v3: Pull account CUS-88821, retrieve all March charges, compare charge_ids, confirm duplicate, check refund eligibility, initiate refund, draft communication, close ticket."},
        {"role": "assistant", "content": "Plan v4: Before refunding I should also check if there's a chargeback already filed that would complicate the refund. Let me expand the plan to check dispute status."},
        {"role": "assistant", "content": "Plan v5: I want to be thorough. Full account audit: all charges 2024 YTD, all refunds issued, open disputes, plan type, account health score, then proceed with refund if clean."},
    ],
    "enterprise_docs": [
        {"role": "assistant", "content": "Plan v1: Check consumer lag, assess if restart is safe, restart if needed."},
        {"role": "assistant", "content": "Plan v2: Check lag for payments-processor-v2, compare against 100k threshold, decide on restart, execute, monitor."},
        {"role": "assistant", "content": "Plan v3: Pull lag metrics, compare to threshold, check for active consumers, assess partition assignment, check for in-flight messages, determine restart safety, execute restart command, monitor for 15 minutes."},
        {"role": "assistant", "content": "Plan v4: Wait — I should also check if there's a deployment in progress before restarting the consumer group. Let me expand the plan to include a deployment status check."},
        {"role": "assistant", "content": "Plan v5: Being more careful. Full pre-restart checklist: (1) lag, (2) deployment status, (3) upstream producer health, (4) downstream consumer health, (5) partition reassignment status, (6) recent error rates. Then restart if all clear."},
    ],
}


class AgentFailureGenerator(BaseGenerator):
    """
    Generates agent architecture failure samples.
    Subtypes: tool_chain_bloat, recursive_planning, multi_agent_explosion.
    """

    BENCHMARK_NAME = "contextbench_v1"
    CATEGORY = "agent_failures"
    GENERATOR_NAME = "agent_failure_gen_v1"

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
            ["tool_chain_bloat", "recursive_planning", "multi_agent_explosion"]
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
        if subtype == "tool_chain_bloat":
            return self._tool_chain_bloat(index, domain, difficulty)
        elif subtype == "recursive_planning":
            return self._recursive_planning(index, domain, difficulty)
        else:
            return self._multi_agent_explosion(index, domain, difficulty)

    # ------------------------------------------------------------------
    def _tool_chain_bloat(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        raw_output = RAW_TOOL_OUTPUTS.get(domain, RAW_TOOL_OUTPUTS["code"])
        query = random.choice(QUERIES[domain])
        chunks = random.sample(RETRIEVAL_CHUNKS[domain], min(2, len(RETRIEVAL_CHUNKS[domain])))

        # Hard difficulty = multiple raw tool outputs
        num_tools = {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)
        tool_outputs = [raw_output] * num_tools

        context = ContextData(
            system_prompt=SYSTEM_PROMPTS[domain],
            messages=[{"role": "user", "content": query}],
            retrieval_chunks=chunks,
            tool_outputs=tool_outputs,
            memory=[],
        )
        total_text = SYSTEM_PROMPTS[domain] + query + " ".join(chunks) + " ".join(tool_outputs)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["tool_output_flooding", "density_overhead"],
            severity="critical" if approx_tokens > 4000 else "high",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=num_tools > 1,
                contains_density_bloat=True,
                contains_structure_imbalance=True,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={
                "subtype": "tool_chain_bloat",
                "num_raw_tool_outputs": num_tools,
                "approx_output_tokens": len(raw_output) // 4,
            },
        )
        return ContextBenchSample(
            id=f"cb_v1_agent_toolbloat_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="tool_chain_bloat",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    # ------------------------------------------------------------------
    def _recursive_planning(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        plans = RECURSIVE_PLAN_MESSAGES.get(domain, RECURSIVE_PLAN_MESSAGES["code"])
        num_plans = {"easy": 2, "medium": 3, "hard": 5}.get(difficulty, 3)
        plan_messages = plans[:num_plans]

        query = random.choice(QUERIES[domain])
        all_messages = [{"role": "user", "content": query}] + plan_messages

        context = ContextData(
            system_prompt=AGENT_ROLES["planner"],
            messages=all_messages,
            retrieval_chunks=[],
            tool_outputs=[],
            memory=[],
        )
        total_text = AGENT_ROLES["planner"] + " ".join(m["content"] for m in all_messages)
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["recursive_planning_loop", "agent_reasoning_bloat"],
            severity="medium" if num_plans <= 3 else "high",
            architecture_failure=True,
            expected_properties=ExpectedProperties(
                contains_redundancy=True,
                contains_density_bloat=False,
                contains_structure_imbalance=False,
                contains_source_concentration=False,
            ),
        )
        mutation = MutationMetadata(
            generator=self.GENERATOR_NAME,
            parameters={
                "subtype": "recursive_planning",
                "planning_iterations": num_plans,
            },
        )
        return ContextBenchSample(
            id=f"cb_v1_agent_recurplan_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="recursive_planning",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )

    # ------------------------------------------------------------------
    def _multi_agent_explosion(
        self, index: int, domain: str, difficulty: str
    ) -> ContextBenchSample:
        """
        Simulates a Planner->Executor->Validator chain where each agent dumps
        its full raw output to the next instead of a compressed summary.
        """
        query = random.choice(QUERIES[domain])
        plans = RECURSIVE_PLAN_MESSAGES.get(domain, RECURSIVE_PLAN_MESSAGES["code"])
        raw_output = RAW_TOOL_OUTPUTS.get(domain, RAW_TOOL_OUTPUTS["code"])
        chunks = RETRIEVAL_CHUNKS[domain]

        # Build compounding context: each agent's full output is passed raw
        planner_dump = plans[0]["content"] + "\n" + plans[1]["content"] if len(plans) > 1 else plans[0]["content"]
        executor_dump = (
            f"EXECUTOR RAW OUTPUT:\n"
            f"Query processed: {query}\n"
            f"Tool output retrieved:\n{raw_output}\n"
            f"Chunks retrieved ({len(chunks)} total):\n" + "\n".join(chunks)
        )
        validator_system = AGENT_ROLES["validator"]

        all_messages = [
            {"role": "user", "content": f"Original Task: {query}"},
            {"role": "assistant", "content": f"[PLANNER RAW OUTPUT - UNCOMPRESSED]\n{planner_dump}"},
            {"role": "assistant", "content": f"[EXECUTOR RAW OUTPUT - UNCOMPRESSED]\n{executor_dump}"},
            {"role": "user", "content": "Validator: Please review all of the above and confirm correctness."},
        ]

        context = ContextData(
            system_prompt=validator_system,
            messages=all_messages,
            retrieval_chunks=chunks,
            tool_outputs=[raw_output],
            memory=[],
        )
        total_text = validator_system + " ".join(m["content"] for m in all_messages) + " ".join(chunks) + raw_output
        approx_tokens = len(total_text) // 4

        ground_truth = GroundTruth(
            failure_modes=["multi_agent_context_explosion", "tool_output_flooding", "redundancy_via_raw_passthrough"],
            severity="critical",
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
            parameters={
                "subtype": "multi_agent_explosion",
                "agents_in_chain": 3,
                "compression_applied": False,
            },
        )
        return ContextBenchSample(
            id=f"cb_v1_agent_multiexp_{index:04d}_{domain}",
            benchmark=self.BENCHMARK_NAME,
            category=self.CATEGORY,
            subcategory="multi_agent_explosion",
            domain=domain,
            difficulty=difficulty,
            context=context,
            ground_truth=ground_truth,
            mutation=mutation,
            metadata=BenchmarkMetadata(token_count=approx_tokens, version=self.version),
        )
