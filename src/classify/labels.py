"""
The paper-type taxonomy: a single source of truth for the 9 canonical labels.

These labels were previously redefined in ~8 places (zero_shot.LABELS keys,
rules.RULES keys, build_dev_set.ALL_CLASSES, evaluate_dev.GOLD_NORMALISE values,
the viz TYPE_ORDER/LABEL_SHORT maps, and the test CANONICAL sets). 
Import from here instead so the vocabulary is defined once.
"""

# Canonical display order: most -> least common in the corpus. The order is used
# for stable legend/stack ordering in the viz; equality checks treat it as a set.
CANONICAL_LABELS: tuple[str, ...] = (
    "Empirical Study",
    "Tool Paper",
    "Case Study",
    "Theoretical Contribution",
    "Survey",
    "Position Paper",
    "Experience Report",
    "Systematic Literature Review",
    "Controlled Experiment",
)

# Order-independent membership/equality check (used by tests and validation).
CANONICAL_SET = frozenset(CANONICAL_LABELS)

# Short forms for compact chart axes and legends. Only the long names are
# abbreviated; the rest are kept verbatim for readability.
SHORT_LABELS: dict[str, str] = {
    "Empirical Study": "Empirical Study",
    "Tool Paper": "Tool Paper",
    "Case Study": "Case Study",
    "Theoretical Contribution": "Theoretical",
    "Survey": "Survey",
    "Position Paper": "Position Paper",
    "Experience Report": "Experience Report",
    "Systematic Literature Review": "SLR",
    "Controlled Experiment": "Controlled Exp.",
}
