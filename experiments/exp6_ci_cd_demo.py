import sys
import os

# Add parent to path to import contextops
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextops.core.models import ContextBundle, ContextItem, ContextType
from contextops.core.engine import analyze

def get_proposed_pipeline() -> ContextBundle:
    """
    Mocks a pipeline configuration that a developer just submitted in a Pull Request.
    In a real project, this would be imported directly from the application's source code.
    
    Scenario: A developer decided to add a massive chunk of Markdown formatting
    to every retrieved document, and heavily duplicated the system prompt instructions.
    """
    
    # A healthy, concise system prompt
    system_prompt = (
        "You are an AI assistant. "
        "Always be polite and respectful. "
        "Answer the query based ONLY on the provided documents."
    )
    
    items = [
        ContextItem(type=ContextType.SYSTEM, content=system_prompt),
        ContextItem(type=ContextType.MESSAGE, content="How does CI/CD work?")
    ]
    
    # Normal retrieved documents
    for i in range(2):
        doc_content = f"CI/CD stands for Continuous Integration and Continuous Deployment. It helps teams release code faster."
        items.append(ContextItem(type=ContextType.RETRIEVAL, content=doc_content, source=f"doc_{i}"))
        
    return ContextBundle(items=items)

def run_ci_cd_check():
    print("==================================================")
    print("ContextOps: CI/CD Pipeline Health Check")
    print("==================================================\n")
    
    MIN_HEALTH_THRESHOLD = 85.0
    
    print("Running static analysis on the proposed RAG payload...")
    pipeline = get_proposed_pipeline()
    
    result = analyze(pipeline)
    score = result.score
    
    print(f"\n[Result] Overall Context Health Score: {score:.1f}/100.0")
    
    if score < MIN_HEALTH_THRESHOLD:
        print("\n[FAIL] CI/CD PIPELINE BLOCKED: Structural Degradation Detected!")
        print(f"The Context Health Score ({score:.1f}) fell below the required threshold ({MIN_HEALTH_THRESHOLD}).")
        
        print("\nDetailed Penalty Breakdown:")
        sb = result.score_breakdown
        if sb.redundancy_penalty > 5:
            print(f"  - Redundancy Penalty: {sb.redundancy_penalty:.1f}/30.0 (Check for duplicated text)")
        if sb.density_penalty > 5:
            print(f"  - Density Penalty: {sb.density_penalty:.1f}/30.0 (Too much Markdown/formatting overhead)")
        if sb.structure_penalty > 5:
            print(f"  - Structure Penalty: {sb.structure_penalty:.1f}/20.0 (Check for overly bloated system prompts)")
        if sb.concentration_penalty > 5:
            print(f"  - Concentration Penalty: {sb.concentration_penalty:.1f}/20.0 (Source collapse)")
            
        print("\nPlease fix the structural inefficiencies before merging this Pull Request.")
        sys.exit(1) # This tells GitHub Actions to fail the build
    else:
        print("\n[PASS] CI/CD PIPELINE PASSED: Context Health is optimal.")
        sys.exit(0)

if __name__ == "__main__":
    run_ci_cd_check()
