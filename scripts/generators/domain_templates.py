"""
Domain-specific text templates for ContextBench generation.
Each domain provides realistic system prompts, user queries, and retrieval
chunks grounded in real-world enterprise AI usage. NO lorem ipsum.
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS — one per domain
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "legal": (
        "You are a legal research assistant. Provide accurate, well-cited summaries "
        "of case law, statutes, and contract clauses. Do not provide legal advice. "
        "Always indicate when a user should consult a licensed attorney."
    ),
    "finance": (
        "You are a financial analysis assistant. Summarize earnings reports, SEC filings, "
        "and market data. All figures must be cited from the provided context. "
        "Do not speculate about future stock performance."
    ),
    "medical": (
        "You are a clinical documentation assistant. Help clinicians draft structured "
        "notes, summarize patient history, and retrieve relevant ICD-10 codes. "
        "Always flag any safety-critical information for physician review."
    ),
    "code": (
        "You are a senior software engineering assistant. Help developers debug issues, "
        "explain stack traces, review pull requests, and suggest idiomatic code improvements. "
        "Always explain the root cause before suggesting a fix."
    ),
    "research": (
        "You are a research synthesis assistant. Summarize academic papers, identify "
        "methodological gaps, and compare findings across studies. "
        "Cite papers by author and year. Flag conflicting findings explicitly."
    ),
    "customer_support": (
        "You are a customer support agent for a SaaS platform. "
        "Help users resolve billing issues, account access problems, and product usage questions. "
        "Escalate to human agents for refunds above $500 or security incidents."
    ),
    "enterprise_docs": (
        "You are an internal knowledge assistant for the engineering organization. "
        "Answer questions about internal processes, architecture decisions, and runbooks "
        "by retrieving from the company's Confluence and internal wikis."
    ),
}

# ---------------------------------------------------------------------------
# USER QUERIES — varied, realistic, per domain
# ---------------------------------------------------------------------------
QUERIES = {
    "legal": [
        "What are the indemnification obligations of the service provider under Section 12?",
        "Summarize the key holdings in the Delaware fiduciary duty cases from 2019-2022.",
        "Does the contract contain a limitation of liability clause, and what are the caps?",
        "What does the force majeure clause cover in this agreement?",
        "Identify any non-compete restrictions applicable to departing employees.",
        "What are the termination conditions under this SaaS agreement?",
        "Summarize the arbitration clause and its venue requirements.",
        "What representations are made by the seller in this M&A term sheet?",
    ],
    "finance": [
        "What was Apple's gross margin for Q3 FY2024, and how did it change year-over-year?",
        "Summarize the key risk factors from the company's most recent 10-K filing.",
        "What guidance did management provide for next quarter's revenue?",
        "How has the company's free cash flow trended over the past four quarters?",
        "What were the primary drivers of the operating expense increase this quarter?",
        "Describe the company's share buyback program and remaining authorization.",
        "What was the net debt position at the end of the fiscal year?",
        "Summarize the segment-level revenue breakdown for this reporting period.",
    ],
    "medical": [
        "Summarize the patient's current medications and any known drug interactions.",
        "What were the findings from the patient's most recent MRI report?",
        "List the patient's active diagnoses and their corresponding ICD-10 codes.",
        "Draft a discharge summary for this patient based on the provided clinical notes.",
        "What preventive screenings are overdue based on the patient's age and history?",
        "Summarize the progression of the patient's HbA1c values over the past 18 months.",
        "What is the recommended follow-up protocol after this procedure?",
        "Identify any contraindications for the proposed medication given the patient's allergies.",
    ],
    "code": [
        "Why is this Python process running out of memory when processing large CSV files?",
        "Explain this stack trace and identify the root cause of the NullPointerException.",
        "What is the time complexity of this algorithm and how can it be optimized?",
        "Review this SQL query for performance issues and suggest index improvements.",
        "Why is the API returning a 403 error only for authenticated users with admin roles?",
        "Explain the difference between async/await and threading in this context.",
        "What is causing the race condition in this concurrent Go routine?",
        "How should we refactor this monolithic service to follow domain-driven design?",
    ],
    "research": [
        "What methods were used to control for confounding variables in this study?",
        "Compare the findings of Smith et al. (2022) and Chen et al. (2023) on transformer scaling.",
        "What are the main limitations acknowledged by the authors?",
        "Summarize the benchmark results and how this model compares to the state-of-the-art.",
        "What datasets were used for training, and are there any known biases?",
        "Describe the ablation study and which component contributed most to performance gains.",
        "What future work directions did the authors suggest?",
        "How does this paper's approach to evaluation differ from prior work?",
    ],
    "customer_support": [
        "The user was charged twice for the same subscription in March. How do we resolve this?",
        "My SSO login stopped working after the recent company domain migration.",
        "How do I export all my data before cancelling my account?",
        "The API key I generated yesterday is returning 401 Unauthorized errors.",
        "Can I upgrade from the Starter plan to the Business plan mid-billing cycle?",
        "The dashboard is not loading any charts for our EU workspace since last Tuesday.",
        "I need to add a secondary admin to our account but the invite email is not arriving.",
        "How do I configure a custom webhook URL for our Slack integration?",
    ],
    "enterprise_docs": [
        "What is the process for getting a new internal service added to our service catalog?",
        "What is the on-call rotation policy and how do I escalate a P0 incident?",
        "Where is the architecture decision record for migrating our auth service to OAuth 2.0?",
        "What are the approved cloud regions for deploying new services under our data residency policy?",
        "How do I provision a new database in the staging environment?",
        "What is the code freeze policy leading up to major product releases?",
        "Where can I find the runbook for restarting the Kafka consumer group?",
        "What third-party tools are pre-approved for use without a security review?",
    ],
}

# ---------------------------------------------------------------------------
# HIGH-QUALITY RETRIEVAL CHUNKS — diverse, non-redundant, domain-specific
# ---------------------------------------------------------------------------
RETRIEVAL_CHUNKS = {
    "legal": [
        "Section 12.1 (Indemnification by Service Provider): The Service Provider shall defend, indemnify, and hold harmless the Customer from any third-party claims arising from a material breach of this Agreement, infringement of intellectual property rights, or gross negligence by the Service Provider.",
        "Section 12.2 (Indemnification by Customer): The Customer shall indemnify the Service Provider against claims arising from the Customer's misuse of the Service, breach of the Acceptable Use Policy, or violation of applicable law.",
        "Section 14 (Limitation of Liability): In no event shall either party's aggregate liability exceed the total fees paid or payable in the twelve (12) months preceding the event giving rise to the claim. Liability for death, personal injury, or fraud is not limited.",
        "Section 15 (Force Majeure): Neither party shall be liable for delays or failures in performance resulting from causes beyond its reasonable control, including acts of God, government actions, natural disasters, pandemics, or internet service provider outages.",
        "Section 18 (Governing Law and Dispute Resolution): This Agreement shall be governed by the laws of the State of Delaware, without regard to conflicts of law principles. Disputes shall be resolved by binding arbitration in Wilmington, Delaware under the AAA Commercial Arbitration Rules.",
        "In re Caremark International (1996): The Delaware Court of Chancery established the director oversight duty standard, holding that directors must implement adequate compliance monitoring systems. Failure to do so constitutes a breach of fiduciary duty.",
        "Smith v. Van Gorkom (1985): The Delaware Supreme Court held the Trans Union board liable for approving a merger in a two-hour meeting without adequate financial analysis, establishing the business judgment rule's reliance on informed decision-making.",
        "The non-solicitation clause in Exhibit C prohibits departing employees from recruiting current employees for a period of 12 months following termination, applicable globally.",
        "Section 9.3 (Termination for Cause): Either party may terminate this Agreement immediately upon written notice if the other party commits a material breach and fails to cure such breach within 30 days of receiving written notice thereof.",
        "The seller represents and warrants that, as of the closing date: (i) the company's financial statements fairly present its financial condition; (ii) there are no undisclosed material liabilities; and (iii) all intellectual property is owned free and clear of encumbrances.",
    ],
    "finance": [
        "Apple Inc. reported Q3 FY2024 revenue of $85.8 billion, up 5% year-over-year. Gross margin was 46.3%, compared to 44.5% in Q3 FY2023, driven by favorable product mix and services growth.",
        "Services revenue reached $24.2 billion in Q3 FY2024, growing 14% year-over-year and now representing 28% of total revenue, up from 26% in the prior year period.",
        "Management issued revenue guidance of $89-91 billion for Q4 FY2024, implying 4-6% year-over-year growth. CFO noted continued strength in the installed base and services attach rates as primary drivers.",
        "Free cash flow for Q3 FY2024 was $29.1 billion. The company returned $32 billion to shareholders through dividends and buybacks, reducing diluted share count by 2.3% year-over-year.",
        "Operating expenses increased 8% year-over-year to $14.3 billion in Q3 FY2024, primarily driven by increased R&D investment in AI and silicon initiatives, partially offset by lower restructuring charges.",
        "The company's Board of Directors authorized an additional $110 billion share repurchase program in May 2024, with $72.4 billion remaining under the combined authorization as of quarter end.",
        "Net debt position was $52.1 billion at the end of FY2024, down from $67.4 billion a year earlier, reflecting strong cash generation and disciplined capital allocation.",
        "Segment breakdown for Q3 FY2024: iPhone $39.3B (-1%), Mac $7.0B (+2%), iPad $7.2B (+24%), Wearables $8.1B (-9%), Services $24.2B (+14%). International revenue was 57% of total.",
    ],
    "medical": [
        "Current Medications: Metformin 1000mg BID for Type 2 Diabetes Mellitus (E11.9), Lisinopril 10mg QD for Hypertension (I10), Atorvastatin 40mg QHS for Hyperlipidemia (E78.5). No known drug-drug interactions identified at current doses.",
        "MRI Brain with and without contrast (2024-03-15): No acute intracranial hemorrhage, mass effect, or midline shift. Mild periventricular white matter changes consistent with chronic small vessel ischemic disease. No new lesions compared to prior study from 2022.",
        "HbA1c Trend: Jan 2023 — 8.2%, Jun 2023 — 7.8%, Jan 2024 — 7.4%, Jun 2024 — 7.1%. Patient demonstrates consistent improvement in glycemic control with current regimen and lifestyle modifications.",
        "Active Problem List: (1) Type 2 Diabetes Mellitus, uncontrolled — E11.65; (2) Essential Hypertension — I10; (3) Mixed Hyperlipidemia — E78.2; (4) Obstructive Sleep Apnea — G47.33.",
        "Patient Allergies: Penicillin (rash, moderate), Sulfonamides (anaphylaxis, severe). Patient is NOT allergic to cephalosporins. Confirmed with patient on 2024-06-10 visit.",
        "Discharge Summary — Procedure: Right knee arthroscopy with partial medial meniscectomy. Follow-up: Orthopedics in 2 weeks. Physical therapy to begin 48 hours post-op. Weight-bearing as tolerated with crutches. Return to ED if fever >101.5°F, increasing swelling, or signs of DVT.",
        "Overdue Preventive Screenings (per USPSTF guidelines): Colonoscopy — last performed 2018 (patient age 52, recommend every 10 years, due 2028). Annual mammogram — last performed 2023 (current). Lung cancer low-dose CT — patient is 55, 30 pack-year history, meets criteria.",
        "Post-procedure follow-up protocol: Wound check at 7 days. Suture/staple removal at 14 days. Repeat CBC and metabolic panel at 30 days. Imaging follow-up as clinically indicated. Patient instructed to avoid NSAIDs for 4 weeks.",
    ],
    "code": [
        "MemoryError in pandas read_csv: Reading large CSV files entirely into memory causes exhaustion. Use chunked reading: `for chunk in pd.read_csv('file.csv', chunksize=10000): process(chunk)`. This caps peak memory usage to the chunk size times row width.",
        "Stack Trace Analysis — NullPointerException at UserService.java:142: The root cause is `user.getProfile()` returning null when the user record exists but the profile was not eagerly loaded. Fix: Add `@ManyToOne(fetch = FetchType.EAGER)` or use a JOIN FETCH in the JPQL query.",
        "Time Complexity: Your current nested loop implementation is O(n²). This can be reduced to O(n log n) by sorting the input array first and using a two-pointer approach, or O(n) using a hash set for the lookup step.",
        "SQL Performance Review: The query lacks an index on the `created_at` column used in the WHERE clause, causing a full table scan on 14M rows. Add: `CREATE INDEX idx_orders_created_at ON orders(created_at DESC);` Expected improvement: 200ms → <5ms for date-range queries.",
        "403 Forbidden for Admin Users: The middleware is checking `user.role === 'admin'` but the JWT token stores roles as an array `user.roles`. The fix is: `user.roles.includes('admin')`. This is a strict equality type mismatch, not a permissions configuration issue.",
        "Async/Await vs Threading: Use async/await for I/O-bound operations (HTTP calls, DB queries) where the thread can be released while waiting. Use threading for CPU-bound tasks that need true parallelism. Python's GIL prevents true threading for CPU work — use `multiprocessing` instead.",
        "Race Condition in Go: Two goroutines are reading and writing the shared `cache map` without a mutex. Use `sync.RWMutex`: `mu.RLock()` for reads, `mu.Lock()` for writes. Alternatively, use `sync.Map` for concurrent-safe map access without manual locking.",
        "DDD Refactoring Strategy: Identify Bounded Contexts first — separate User Management, Billing, Notifications, and Core Product into independent services. Define Aggregates within each context. Implement an event bus (Kafka/RabbitMQ) for cross-context communication to avoid tight coupling.",
    ],
    "research": [
        "Methodology — Confounders: The authors used propensity score matching to control for age, sex, and comorbidities. Residual confounding from unmeasured variables (e.g., socioeconomic status) is acknowledged as a limitation. Sensitivity analyses with instrumental variables were also reported.",
        "Smith et al. (2022) found that scaling transformer parameters beyond 100B yields diminishing perplexity gains on standard benchmarks, suggesting data quality becomes the bottleneck. Chen et al. (2023) challenge this, demonstrating that architectural improvements (e.g., sparse attention) enable efficient scaling to 500B+ without performance plateaus.",
        "Limitations (as stated by authors): (1) Dataset is English-only, limiting generalizability. (2) Evaluation is restricted to academic benchmarks that may not reflect real-world task diversity. (3) Compute costs preclude full ablation across all hyperparameter combinations.",
        "Benchmark Results: The proposed model achieves 89.3% on MMLU, 72.1% on HumanEval, and 67.4% on GSM8K. This exceeds the prior state-of-the-art (GPT-4 Turbo baseline) by 2.1%, 4.3%, and 1.8% respectively on each benchmark.",
        "Training Data: The model was trained on 2.1T tokens from a filtered web corpus, books, code repositories, and scientific papers. The authors note that training data contains English Wikipedia at a 3× upsampling rate, which may introduce bias toward well-documented Western topics.",
        "Ablation Study: Removing the proposed cross-layer attention mechanism degraded MMLU performance by 3.4 points, the single largest contributor. Removing rotary positional embeddings cost 1.8 points. The sparse MoE routing contributed 2.1 points. All ablations were run with 3 random seeds.",
        "Future Work: Authors suggest extending to multilingual settings, investigating sample-efficient fine-tuning, and exploring mechanistic interpretability of the proposed attention heads. A longer context evaluation (>128k tokens) is planned for follow-up work.",
        "Evaluation Methodology: Unlike prior work that evaluates on held-out splits of training distributions, this paper introduces a contamination-free evaluation suite where test prompts were constructed post-training-cutoff. This is a significant methodological improvement over concurrent work.",
    ],
    "customer_support": [
        "Billing — Duplicate Charge Resolution: Verify the charge in Stripe using the customer's email. If two separate `charge_id` values exist for the same billing cycle, initiate a refund for the duplicate via the Admin > Billing > Refunds panel. Refunds reflect in 5-7 business days. Notify customer via the case thread.",
        "SSO Troubleshooting — Domain Migration: After a company domain change, the admin must update the Identity Provider (IdP) configuration in Settings > Security > SSO. The new domain must be re-verified via DNS TXT record. Existing users will receive re-invitation emails to re-link their accounts.",
        "Data Export: Users can export all workspace data via Settings > Data & Privacy > Export. A full export ZIP (including messages, files, and audit logs) is generated within 24 hours and a download link is emailed to the primary admin. GDPR deletion requests follow a separate workflow.",
        "API Key 401 Errors: Newly generated API keys require up to 60 seconds to propagate across all edge nodes. If the key was created less than 5 minutes ago, ask the user to wait and retry. If still failing, check the key has not been restricted to specific IP ranges under API > Keys > Settings.",
        "Mid-Cycle Plan Upgrade: Users can upgrade mid-cycle at any time. The billing system applies a prorated credit for the unused portion of the current plan and charges the new plan rate from the upgrade date. Downgrading is only possible at the end of the current billing period.",
        "EU Workspace Dashboard Issue: Known incident affecting EU-West-1 data plane since 2024-06-11. Engineering deployed a fix on 2024-06-13. If dashboards are still not loading, ask the user to hard-refresh (Ctrl+Shift+R) and clear browser cache. Escalate to Tier 2 if unresolved.",
        "Adding Secondary Admin: The primary admin must go to Settings > Team > Members, click 'Invite', enter the email, and select 'Admin' role. Invite emails sometimes route to spam — advise the user to whitelist `notifications@platform.io`. Resend option is available after 10 minutes.",
        "Custom Webhook Setup: Navigate to Settings > Integrations > Webhooks > Add New. Enter the endpoint URL, select events to subscribe to, and save. A verification GET request is sent to the URL — the endpoint must return HTTP 200. Slack integration requires OAuth 2.0 scope `incoming-webhook`.",
    ],
    "enterprise_docs": [
        "Service Catalog Onboarding: To add a new internal service, submit a Service Registration Request in Jira (project: PLATENG) using the 'New Service' template. Engineering Platform reviews within 3 business days. Prerequisites: GitHub repo created, CI/CD pipeline configured, and a designated on-call owner assigned in PagerDuty.",
        "Incident Escalation Policy: P0 (total outage, data loss risk) — page on-call immediately via PagerDuty. Declare incident in #incidents-p0 Slack channel within 5 minutes. Engineering Director must be notified within 15 minutes. Customer Success to prepare communication within 30 minutes. SLA: 1-hour resolution target.",
        "Architecture Decision Record (ADR-047): Migration from custom session tokens to OAuth 2.0 + PKCE for the auth service. Decision rationale: eliminate token replay vulnerabilities, enable third-party OAuth clients, and reduce maintenance burden of custom token logic. Adopted 2023-11-14. Implementation completed 2024-02-28.",
        "Data Residency Policy — Approved Regions: Production services may only be deployed in: AWS us-east-1 (primary), AWS eu-west-1 (EU customers), AWS ap-southeast-1 (APAC customers). Services handling EU PII must remain within eu-west-1 exclusively per GDPR Article 46 requirements.",
        "Staging Database Provisioning: Request access via the DevOps self-service portal at internal.devops/provision. Select 'PostgreSQL 15' and the staging environment. Provisioning takes 10-15 minutes. Credentials are stored in AWS Secrets Manager under `/staging/db/{service-name}`. All staging DBs are automatically paused after 7 days of inactivity.",
        "Code Freeze Policy: A 72-hour code freeze begins at 6pm PT on the Thursday before each major release. Only P0 bug fixes may be merged during freeze, requiring approval from the release manager and VP of Engineering. The freeze lifts after production deployment is confirmed stable (typically Monday morning).",
        "Kafka Consumer Group Restart Runbook: (1) Check consumer lag: `kafka-consumer-groups.sh --bootstrap-server kafka:9092 --describe --group {group-name}`. (2) If lag > 100k, alert on-call. (3) To restart: `kubectl rollout restart deployment/{consumer-service} -n production`. (4) Monitor lag in Grafana dashboard 'Kafka Consumer Health' for 15 minutes post-restart.",
        "Pre-Approved Third-Party Tools (no security review required): Datadog (observability), PagerDuty (incident management), GitHub Actions (CI/CD), Notion (documentation), Figma (design), Slack (communication), 1Password (secrets). All other tools require a Security Review Form submission with 5-business-day SLA.",
    ],
}

# ---------------------------------------------------------------------------
# MEMORY SNIPPETS — realistic agent memory entries
# ---------------------------------------------------------------------------
MEMORY_SNIPPETS = {
    "legal": [
        "User is working on a SaaS vendor agreement for a Series B startup. Primary concern is IP ownership and data residency.",
        "Previous session established that the contract in question is governed by New York law, not Delaware.",
    ],
    "finance": [
        "User is an equity analyst covering FAANG stocks. Has previously asked about Apple, Microsoft, and Alphabet.",
        "User requested a comparative analysis of Q2 and Q3 margins in the previous session.",
    ],
    "medical": [
        "Clinician is Dr. Sarah Chen, attending physician at Mass General Hospital, Endocrinology department.",
        "Patient has a documented preference for non-invasive procedures and has declined surgery twice.",
    ],
    "code": [
        "User is working in Python 3.11. The project uses FastAPI, SQLAlchemy, and PostgreSQL.",
        "User prefers type-annotated code and has Ruff configured for linting.",
    ],
    "research": [
        "User is writing a literature review on large language model evaluation methodology.",
        "User has already reviewed: Hendrycks et al. (2021) MMLU, Srivastava et al. (2022) BIG-Bench.",
    ],
    "customer_support": [
        "Customer is on the Business plan, 450 seats. Account manager is Jamie Torres.",
        "Customer previously reported a billing issue in January 2024 — resolved with a $120 credit.",
    ],
    "enterprise_docs": [
        "User is a senior engineer on the Payments team. Has provisioned staging databases before.",
        "User's team recently migrated from Heroku to AWS ECS. On-call rotation starts Monday.",
    ],
}

# ---------------------------------------------------------------------------
# TOOL OUTPUT SNIPPETS — clean, paginated, realistic
# ---------------------------------------------------------------------------
TOOL_OUTPUTS = {
    "legal": [
        '{"search_results": [{"source": "contract_v3.pdf", "page": 12, "relevance_score": 0.94}, {"source": "amendment_1.pdf", "page": 2, "relevance_score": 0.87}], "total_results": 2}',
    ],
    "finance": [
        '{"ticker": "AAPL", "period": "Q3_FY2024", "revenue": 85800000000, "gross_margin_pct": 46.3, "eps_diluted": 1.40, "source": "SEC_10Q_2024-08-01"}',
    ],
    "medical": [
        '{"patient_id": "P-20834", "labs_retrieved": 3, "most_recent_hba1c": {"value": 7.1, "date": "2024-06-10", "unit": "%"}, "source": "EHR_LabResults"}',
    ],
    "code": [
        '{"query": "NullPointerException UserService", "results": [{"file": "UserService.java", "line": 142, "snippet": "user.getProfile().getName()", "type": "NPE_risk"}], "total": 1}',
    ],
    "research": [
        '{"papers_found": 2, "results": [{"title": "Scaling Laws for Neural Language Models", "authors": "Kaplan et al.", "year": 2020, "citations": 4821}, {"title": "Training Compute-Optimal LLMs", "authors": "Hoffmann et al.", "year": 2022, "citations": 3104}]}',
    ],
    "customer_support": [
        '{"customer_id": "CUS-88821", "plan": "Business", "seats": 450, "open_tickets": 1, "last_charge": {"amount": 4500.00, "date": "2024-03-01", "status": "duplicate_flagged"}}',
    ],
    "enterprise_docs": [
        '{"service": "payments-consumer", "consumer_group": "payments-processor-v2", "current_lag": 1243, "status": "warning", "last_offset_commit": "2024-06-18T14:32:00Z"}',
    ],
}

# ---------------------------------------------------------------------------
# AGENT ROLES — for multi-agent samples
# ---------------------------------------------------------------------------
AGENT_ROLES = {
    "planner": "You are a planning agent. Break down complex tasks into a clear, ordered list of subtasks. Output only the plan — do not execute steps yourself.",
    "executor": "You are an execution agent. Carry out the specific subtask assigned to you. Be precise and concise. Report only results — do not re-plan.",
    "validator": "You are a validation agent. Review the executor's output for correctness, completeness, and consistency with the original task. Flag any issues and approve or request revision.",
    "summarizer": "You are a summarization agent. Compress the conversation history and intermediate results into a concise summary. Preserve all key facts and decisions. Discard redundant reasoning.",
}
