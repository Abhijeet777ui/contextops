import pytest
import random
from contextops.analyzers.redundancy import analyze_redundancy
from contextops.core.models import ContextBundle, ContextItem, ContextType

def test_lsh_mathematical_guarantees():
    """
    Test that 30 bands of 3 hashes provide the expected collision probabilities.
    s=0.5 -> ~98% recall
    s=0.1 -> ~0% false positive
    """
    def simulate_collision(s: float, b: int=30, r: int=3) -> bool:
        sig_a = []
        sig_b = []
        for i in range(b * r):
            val = random.randint(1, 1000000)
            sig_a.append(val)
            if random.random() < s:
                sig_b.append(val)
            else:
                sig_b.append(random.randint(2000000, 3000000))
        
        # Banding check
        for band_idx in range(b):
            if sig_a[band_idx*r:(band_idx+1)*r] == sig_b[band_idx*r:(band_idx+1)*r]:
                return True
        return False
        
    # Recall at s=0.5 should be ~98%
    matches_50 = sum(1 for _ in range(1000) if simulate_collision(0.5))
    recall_rate = matches_50 / 1000.0
    assert 0.95 <= recall_rate <= 1.0, f"Recall at s=0.5 was {recall_rate}, expected ~0.98"
    
    # False positive at s=0.1 should be ~0%
    matches_10 = sum(1 for _ in range(1000) if simulate_collision(0.1))
    fp_rate = matches_10 / 1000.0
    assert 0.0 <= fp_rate <= 0.05, f"False positive at s=0.1 was {fp_rate}, expected ~0.0"

def test_token_inverted_index_strict_semantic():
    bundle = ContextBundle(items=[
        ContextItem(id="1", type=ContextType.RETRIEVAL, content="apple banana cherry date elderberry fig grape", token_count=7),
        ContextItem(id="2", type=ContextType.RETRIEVAL, content="date elderberry fig honeydew kiwi lemon mango", token_count=7),
        ContextItem(id="3", type=ContextType.RETRIEVAL, content="apple banana cherry date elderberry fig grape", token_count=7),
    ])
    
    # Without strict_semantic, LSH is used.
    findings_lsh, _ = analyze_redundancy(bundle, strict_semantic=False)
    
    # With strict_semantic, Token Inverted Index is used.
    findings_exact, _ = analyze_redundancy(bundle, strict_semantic=True)
    assert len(findings_exact) >= 1
    
    found_apple = False
    for f in findings_exact:
        if (f.item_a_id == "1" and f.item_b_id == "3") or (f.item_a_id == "3" and f.item_b_id == "1"):
            found_apple = True
    assert found_apple, "Token Inverted Index must catch the exact overlap"

def test_lsh_real_text_collision():
    """
    Spot-check against real text pairs to ensure LSH generates candidates
    without relying on synthetic signature generation.
    """
    text_a = "The quick brown fox jumps over the lazy dog. " * 5  # Ensure > 50 chars for MinHash
    text_b = "The quick brown fox leaps over the sleeping dog. " * 5  # Highly similar but not identical
    text_c = "A completely unrelated document about database sharding and scaling out. " * 5
    
    bundle = ContextBundle(items=[
        ContextItem(id="a", type=ContextType.RETRIEVAL, content=text_a, token_count=50),
        ContextItem(id="b", type=ContextType.RETRIEVAL, content=text_b, token_count=50),
        ContextItem(id="c", type=ContextType.RETRIEVAL, content=text_c, token_count=50),
    ])
    
    # Run through full redundancy pipeline (which uses LSH internally).
    findings, _ = analyze_redundancy(bundle, strict_semantic=False)
    
    # A and B should be caught as highly redundant.
    found_ab = any(f for f in findings if {f.item_a_id, f.item_b_id} == {"a", "b"})
    # A/C and B/C should be completely ignored due to near-zero overlap.
    found_ac = any(f for f in findings if "c" in {f.item_a_id, f.item_b_id})
    
    assert found_ab, "LSH failed to catch real-text high overlap pair."
    assert not found_ac, "LSH falsely caught completely unrelated pair."
