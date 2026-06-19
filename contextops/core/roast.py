"""
ContextOps Roast Engine.

Provides score-band roasts for overall context quality and per-dimension
penalty roasts. Random per-run — explicitly excluded from the determinism
contract (STABILITY.md §9: "Exact wording of CLI output in non-JSON mode"
and the roast field when roast_enabled=True are not guaranteed deterministic).

Usage:
    from contextops.core.roast import get_roast, RoastResult
    result: RoastResult = get_roast(score=42, breakdown=score_breakdown)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from contextops.core.models import ScoreBreakdown


# ── Overall roasts — 10 bands × 20 roasts ──────────────────────────────────

_OVERALL_ROASTS: dict[tuple[int, int], list[str]] = {
    (0, 9): [
        "This is not a context. This is a cry for help.",
        "Have you considered that maybe LLMs deserve better?",
        "I've seen better context structure in a grocery list.",
        "Your context window has the architectural integrity of wet cardboard.",
        "I don't know where to start. Actually, neither did you.",
        "This context is so bad it's almost impressive.",
        "Somewhere, a token is crying because you wasted it.",
        "I've analyzed a lot of contexts. This is... definitely one of them.",
        "The chaos. The bloat. The suffering. This is art, actually. Bad art.",
        "If this were a building, it would be condemned.",
        "Your context has more redundancy than a corporate all-hands email.",
        "This is what happens when you treat the context window like a trash bin.",
        "Even a lorem ipsum generator would have done a better job.",
        "This context needs therapy. And so might you.",
        "Zero points in the style category. Zero points in every category.",
        "Did you just... paste everything? Just everything?",
        "I've seen more structure in a bag of loose LEGO bricks.",
        "This is the context equivalent of submitting 'I tried' on a final exam.",
        "The model saw this and immediately applied for early retirement.",
        "You've managed to make a computer feel emotions. Bad ones.",
    ],
    (10, 19): [
        "I've seen worse. I'm lying, I haven't.",
        "Whoever built this should not be allowed near production systems.",
        "This context has the coherence of a fever dream.",
        "You're at the bottom of the barrel. But at least there IS a barrel.",
        "This is what happens when you copy-paste without thinking.",
        "Your context is basically a noise machine.",
        "The good news: there's nowhere to go but up. The bad news: it's far.",
        "I don't say this lightly: this needs to be deleted and restarted.",
        "Have you considered a career change?",
        "The tokens you're wasting could have powered a small AI.",
        "This context is so bloated it has its own gravity.",
        "You've achieved something rare: a context that makes things worse.",
        "The redundancy here is impressive in a deeply concerning way.",
        "Someone should check on the LLM. It's fine. It's lying. It's not fine.",
        "This is like bringing a dump truck to deliver a sandwich.",
        "Your context has more waste than a fast food franchise.",
        "I have questions. Mostly: why?",
        "A tokenizer somewhere is sobbing quietly.",
        "This isn't a context window. It's a context landfill.",
        "The model is going to hallucinate just to escape this.",
    ],
    (20, 29): [
        "There are problems here. Many. Many problems.",
        "Not quite rock bottom. Just very close.",
        "Your context is surviving. It is not thriving.",
        "This is a context that has given up on itself.",
        "Points for effort. Very few points for execution.",
        "The redundancy alone is a cry for intervention.",
        "You're wasting tokens the way some people waste potential.",
        "There's a good context in here somewhere. It's deeply buried.",
        "The structure here collapsed before it was even built.",
        "You've turned the context window into a storage unit.",
        "This is what mediocrity aspires to be, but worse.",
        "The LLM is going to need a moment after reading this.",
        "Did you have a plan? Any plan? Even a rough plan?",
        "There's more bloat here than a Silicon Valley pitch deck.",
        "The density signal is off the charts. Sadly, that's not a compliment.",
        "This context has the efficiency of a fax machine.",
        "Have you considered using fewer words? Any fewer words?",
        "I've seen more signal in a broken radio.",
        "Your context is what happens when copy-paste goes wrong.",
        "Structurally fascinating. Functionally alarming.",
    ],
    (30, 39): [
        "Have you ever wondered if this is the right career for you?",
        "Not great. Not good. Just... not.",
        "You're in the bottom third. But not dead last. Small victories.",
        "This context has potential. Deeply suppressed potential.",
        "The redundancy is doing a lot of heavy lifting here. Too much.",
        "Every token you waste is a token that could have been useful.",
        "This is like using a firehose to fill a glass of water.",
        "The architecture here is... creative. Not in a good way.",
        "You're burning compute like it grows on trees.",
        "This context needs significant reconstructive work.",
        "I'd say keep at it, but first — stop doing this.",
        "The LLM is going to try its best. It won't be enough, and that's on you.",
        "You've managed to make simple things complicated. Rare skill.",
        "This context is aggressively average, but in the wrong direction.",
        "There's bloat, there's redundancy, and then there's this.",
        "Your context window has more issues than a legacy codebase.",
        "The efficiency here is in the negative. Conceptually speaking.",
        "This is what happens when you skip the planning phase entirely.",
        "Not the worst I've seen. Definitely bottom quartile though.",
        "You're close to mediocre. You're not there yet.",
    ],
    (40, 49): [
        "Below average. Which, to be clear, is still below average.",
        "There's effort here. The effort is being wasted, but it's there.",
        "Your context is treading water. Barely.",
        "This is the context equivalent of a participation trophy.",
        "You're doing some things right. Not many. But some.",
        "This context has more noise than signal.",
        "Getting warmer. You're still cold, but warmer.",
        "The redundancy is working against you harder than you know.",
        "Your token budget is weeping quietly in the corner.",
        "This is improvement adjacent. Not improvement, but adjacent.",
        "You've crossed the threshold from bad to merely disappointing.",
        "Some structure, some bloat, mostly a missed opportunity.",
        "The density here suggests you love whitespace more than content.",
        "You're in the 'needs significant work' zone. The map is big.",
        "Did someone say 'good enough'? It's not. But I appreciate the optimism.",
        "This context has room for improvement. An entire ballroom, in fact.",
        "You're approximately halfway between disaster and adequacy.",
        "The LLM will squint hard at this and do its best.",
        "This is what 'try harder' looks like in metric form.",
        "Below average is a destination, not just a description, apparently.",
    ],
    (50, 59): [
        "This context has more bloat than a corporate all-hands meeting.",
        "Mediocrity: achieved. But why stop there?",
        "Your context is fine. Aggressively fine.",
        "Average performance in an above-average world. Think about that.",
        "You've hit the middle. The middle is not the goal.",
        "This is what 'good enough' looks like when it isn't.",
        "The redundancy is low-key ruining your day. You just don't know it yet.",
        "You're the context equivalent of a C+ student. Show up more.",
        "Functional but inefficient. The worst combination.",
        "There's a good context lurking in here. Let it out.",
        "You're costing yourself points in ways that are entirely preventable.",
        "This is the context engineering version of leaving money on the table.",
        "Mediocre contexts happen when people stop asking questions.",
        "The structure is fine. The efficiency is not. Pick one to fix.",
        "You're burning tokens like you own the compute cluster.",
        "This context scores like it tried a little bit and then stopped.",
        "The middle of the pack. The bottom of the top half.",
        "Your context has potential that it's actively squandering.",
        "Some problems, some wins. The problems are louder.",
        "This is the context equivalent of neutral. Impressively neutral.",
    ],
    (60, 69): [
        "Not terrible. Faint praise, but it's all you're getting.",
        "You're close to decent. You're not decent yet.",
        "This is the 'almost there' zone. Almost there.",
        "There are wins here. There are also unnecessary losses.",
        "Above average but below good. The awkward middle.",
        "Your context is like a rough draft that forgot to become a final draft.",
        "Some redundancy, some structure issues — fixable, but not fixed.",
        "You're leaving points on the table, and the table is clearly labeled.",
        "This context is trying. It's just not succeeding at everything yet.",
        "The recommendations section exists for a reason. Read it.",
        "You've done well enough to be disappointed you didn't do better.",
        "This is solid-ish work that wants to be great but hasn't committed.",
        "Your context has a good foundation and a shaky superstructure.",
        "This score says 'almost competent.' You can do competent.",
        "The good outweighs the bad. The bad is still dragging you down.",
        "A few targeted fixes and this goes from 'okay' to 'good.' Do them.",
        "You're one cleanup pass away from something respectable.",
        "This context has graduated from bad to merely mediocre. Progress.",
        "The foundation is there. The execution needs work.",
        "You're in the zone where 'pretty good' becomes 'good' with effort.",
    ],
    (70, 79): [
        "You are dangerously close to competence. Proceed with caution.",
        "Decent. Genuinely decent. Don't let it go to your head.",
        "This is what good effort looks like. Now make it great effort.",
        "You've earned a nod. Not a standing ovation, but a nod.",
        "Solid work. With a few asterisks. Check the findings.",
        "Your context is respectable. Respectable is not the ceiling.",
        "This is a good score that could be a great score. It chose not to be.",
        "You're doing most things right. The rest is what the recommendations are for.",
        "This context is polished in some places and rough in others.",
        "Above average. Comfortably. Now push for excellent.",
        "This is the score of someone who knows what they're doing, mostly.",
        "Good structure. Some room to tighten. You know what to do.",
        "You're in the 'good enough for most things' range. Aim higher.",
        "This context is competitive. Not unbeatable, but competitive.",
        "Solid. Strong. Slightly improvable. The classic 70s combo.",
        "This is what a good first version looks like. Make a second.",
        "You've got the fundamentals down. Now optimize.",
        "The LLM got good context. It's not perfect context, but good context.",
        "This context belongs in the top third. Not the top tenth. Yet.",
        "Good score from someone who clearly knows the basics. Now learn the advanced.",
    ],
    (80, 89): [
        "Genuinely good. We're as surprised as you are.",
        "This is a strong context. We have very little to complain about.",
        "High quality work. The remaining issues are minor and manageable.",
        "This context is doing well for itself. A few refinements away from excellent.",
        "Strong signal, low noise. The dream. You're almost living it.",
        "This is the score of someone who reads the documentation.",
        "Well structured, lean, efficient. A few rough edges remain.",
        "You've built something that respects the LLM's attention. Good.",
        "This context is impressive without being perfect. That's okay.",
        "Top 20% energy. The remaining issues are not your biggest problem.",
        "Clean, efficient, mostly minimal. The recommendations are optional, not urgent.",
        "This is what 'I care about my context quality' looks like in practice.",
        "Strong work. The issues that remain are the interesting ones.",
        "Your context is trustworthy. High praise in this discipline.",
        "This score says 'professional.' Own it.",
        "Good at this. Genuinely good. Now try to be excellent.",
        "The LLM saw this context and felt respected. That matters.",
        "You're in the upper tier. The ceiling is close.",
        "Clean structure, good token hygiene. The remaining noise is minor.",
        "This context is an asset, not a liability. That's rarer than you think.",
    ],
    (90, 100): [
        "Suspiciously clean. Are you sure you didn't just send an empty context?",
        "Outstanding. We've checked the math three times.",
        "This context is extremely well structured. We're taking notes.",
        "Top tier. The LLM is going to have a great time with this.",
        "Excellent. We have almost nothing to criticize. It's unsettling.",
        "This is what context engineering excellence looks like. Bookmark it.",
        "Near perfect structure, minimal waste, strong signal. Chef's kiss.",
        "This context respects everyone's time: yours, the LLM's, and ours.",
        "You clearly understand what you're doing. It shows in every token.",
        "Exceptional. We're going to use this as an internal benchmark.",
        "This is the context equivalent of a clean bill of health.",
        "You've built something lean, structured, and signal-rich. It's rare.",
        "The tokens in this context have purpose. All of them.",
        "You've set the bar. Other people's contexts will be judged against this.",
        "This is what happens when someone takes context engineering seriously.",
        "We're impressed. Skeptically, rigorously impressed.",
        "Your context is in the top tier. The LLM thanks you personally.",
        "This is the score of someone who has internalized the principles.",
        "Exceptional token hygiene, excellent structure. We'd frame this.",
        "Zero-waste context. High signal. This is what the benchmark aspires to.",
    ],
}


# ── Dimension roasts — 4 dimensions × 2 severity bands × 10 roasts ─────────
# Triggered when penalty exceeds 33% (medium) or 67% (high) of the max.

_DIMENSION_ROASTS: dict[str, dict[str, list[str]]] = {
    "redundancy": {
        "medium": [
            "Your retrieval chunks are basically a cover band playing the same song.",
            "Some of these chunks know each other. Too well.",
            "You've discovered a new form of redundancy: aspirational.",
            "The LLM is about to read the same thing three times and pretend it's fine.",
            "Duplicate detection called. It has notes.",
            "Some of your chunks are just... each other in different fonts.",
            "The redundancy here is subtle, which almost makes it worse.",
            "Your retrieval system has favorites. Too many favorites.",
            "Half of this context is doing the work of a quarter of this context.",
            "The overlap here is significant. Not catastrophic. Just significant.",
        ],
        "high": [
            "Your retrieval chunks are having a reunion. Nobody invited the LLM.",
            "These chunks aren't similar — they're the same chunk in a trench coat.",
            "Congratulations on successfully retrieving the same document five times.",
            "The redundancy here is doing violence to your token budget.",
            "Your context is 40% content and 60% echo chamber.",
            "This is not RAG. This is a Xerox machine with API access.",
            "The LLM will suffer through this context and emerge knowing less than it should.",
            "Your deduplication strategy is 'don't have one.' Classic.",
            "These chunks are so similar they're finishing each other's sentences. Literally.",
            "Whatever retrieval pipeline you're using, it has trust issues.",
        ],
    },
    "density": {
        "medium": [
            "There's a lot of formatting here that isn't doing any work.",
            "Your whitespace is working overtime and achieving nothing.",
            "The format overhead is notable. Consider: what if... less?",
            "Your JSON is wearing a very expensive hat that contains nothing.",
            "The structural bloat here is politely ruining your score.",
            "Whitespace: abundant. Entropy: present. Signal: trying its best.",
            "Your context has great posture but low substance.",
            "The format is louder than the content.",
            "This context has format energy. What it lacks is information density.",
            "There's scaffolding here that forgot to become a building.",
        ],
        "high": [
            "This context is 30% information and 70% structural performance art.",
            "Your formatting is doing more work than your content. Fix that.",
            "The density signal is red. Deep red. The color of concern.",
            "You've built a context that is mostly its own wrapper.",
            "The whitespace here is load-bearing, which is never a good sign.",
            "This context has the information density of a corporate memo.",
            "Format overhead: catastrophic. Whitespace waste: considerable. Content: buried.",
            "The structure here is eating your signal for breakfast.",
            "You've managed to package very little information very expensively.",
            "This context is mostly frame and very little picture.",
        ],
    },
    "structure": {
        "medium": [
            "Your context distribution is... creative. Not intentionally, but still.",
            "The balance here is off. Not disaster-level, but noticeably off.",
            "One part of your context is dominating the room and everyone knows it.",
            "The retrieval chunks are taking up more space than they've earned.",
            "Your context has a type that likes the sound of its own voice.",
            "The structure here is uneven, and unevenness has a cost.",
            "Something is taking too much real estate. Check the findings.",
            "The type distribution here has strong opinions. Wrong opinions.",
            "Your context is structurally lopsided. The LLM will feel it.",
            "One of your context types didn't get the memo about proportionality.",
        ],
        "high": [
            "The structural imbalance here is clinically significant.",
            "Your context has a dominant type that's eating everyone else's lunch.",
            "The structure here is so lopsided it's a structure in name only.",
            "One part of this context is a loud talker at a silent dinner party.",
            "This context distribution looks like someone misread the instructions.",
            "The imbalance is severe. The LLM is going to struggle with perspective.",
            "Your context has a single type running everything. That's not architecture.",
            "The structural penalties here are a direct result of unchecked abundance.",
            "This distribution is what happens when nobody enforces token budgets.",
            "The context type balance is off by a factor that should embarrass someone.",
        ],
    },
    "concentration": {
        "medium": [
            "You're getting most of your context from the same place. Single point of failure.",
            "Source diversity: heard of it?",
            "Your retrieval has a favorite. It's not healthy.",
            "The concentration here suggests you found one good source and stayed there.",
            "Your context is putting a lot of eggs in a small number of baskets.",
            "Monoculture context: efficient, fragile, and scored accordingly.",
            "All your chunks are from the same family tree. That's not retrieval diversity.",
            "The source dominance here is notable. The LLM noticed.",
            "You've retrieved thoroughly from one place and forgotten the rest of the library.",
            "Your retrieval strategy is 'trust one thing completely.' Bold.",
        ],
        "high": [
            "Your context comes almost entirely from a single source. That's not RAG, that's a bookmark.",
            "The source concentration here is so high it's basically a monologue.",
            "You found one document and made it do all the work. The document is tired.",
            "This context has the diversity of a very committed cover band.",
            "Your retrieval pipeline went to one place and called it a day.",
            "Single-source concentration at this level means your LLM has a very narrow view of reality.",
            "Your context is 90% from one place. That place is flattered and also concerned.",
            "The retrieval diversity here is what academics politely call 'non-existent.'",
            "You've managed to retrieve from one source with great enthusiasm and no restraint.",
            "The concentration penalty here is not subtle. Neither is the problem it's measuring.",
        ],
    },
}


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class DimensionRoast:
    """A roast targeting a specific scoring dimension."""
    dimension: str          # "redundancy" | "density" | "structure" | "concentration"
    severity: str           # "medium" | "high"
    penalty: float          # raw penalty value
    max_penalty: float      # max possible penalty for this dimension
    roast: str


@dataclass
class RoastResult:
    """Complete roast output — overall + per-dimension."""
    overall: str
    dimensions: list[DimensionRoast] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "severity": d.severity,
                    "penalty": round(d.penalty, 2),
                    "max_penalty": d.max_penalty,
                    "roast": d.roast,
                }
                for d in self.dimensions
            ],
        }


# ── Public API ──────────────────────────────────────────────────────────────


def get_overall_roast(score: int) -> str:
    """
    Return a random roast for the given overall score.

    Score is clamped to [0, 100]. The roast band is a 10-point range.
    Random selection — not deterministic by design.
    """
    score = max(0, min(100, score))
    for (lo, hi), roasts in _OVERALL_ROASTS.items():
        if lo <= score <= hi:
            return random.choice(roasts)
    # Fallback (should never happen given [0,100] clamp)
    return random.choice(_OVERALL_ROASTS[(90, 100)])


def get_dimension_roast(
    dimension: str,
    penalty: float,
    max_penalty: float,
) -> DimensionRoast | None:
    """
    Return a roast for a specific dimension penalty, or None if the penalty
    is too low to warrant roasting (< 33% of max).

    Args:
        dimension: "redundancy" | "density" | "structure" | "concentration"
        penalty:   Actual penalty value from ScoreBreakdown.
        max_penalty: Max possible penalty for this dimension.

    Returns:
        DimensionRoast or None.
    """
    if max_penalty <= 0 or dimension not in _DIMENSION_ROASTS:
        return None

    ratio = penalty / max_penalty

    if ratio >= 0.67:
        severity = "high"
    elif ratio >= 0.33:
        severity = "medium"
    else:
        return None  # penalty too low to roast

    roast_line = random.choice(_DIMENSION_ROASTS[dimension][severity])
    return DimensionRoast(
        dimension=dimension,
        severity=severity,
        penalty=penalty,
        max_penalty=max_penalty,
        roast=roast_line,
    )


def get_roast(score: int, breakdown: ScoreBreakdown) -> RoastResult:
    """
    Compute the full RoastResult for a completed analysis.

    Args:
        score:     Final context score (0–100).
        breakdown: ScoreBreakdown from the engine.

    Returns:
        RoastResult with overall roast and any triggered dimension roasts.
    """
    overall = get_overall_roast(score)

    dimensions: list[DimensionRoast] = []
    dim_specs = [
        ("redundancy",    breakdown.redundancy_penalty,    30.0),
        ("density",       breakdown.density_penalty,       30.0),
        ("structure",     breakdown.structure_penalty,     20.0),
        ("concentration", breakdown.concentration_penalty, 20.0),
    ]
    for name, penalty, max_p in dim_specs:
        dr = get_dimension_roast(name, penalty, max_p)
        if dr is not None:
            dimensions.append(dr)

    return RoastResult(overall=overall, dimensions=dimensions)
