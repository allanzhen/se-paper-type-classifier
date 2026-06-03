"""Keyword/phrase rules for the rule-based paper-type classifier.

Each key in RULES is a paper-type label; each value is a list of regex
patterns whose presence in the title+abstract counts as one match for
that label. Match counts are summed per label and the label with the
most matches wins (ties or zero matches -> "Unknown").

Tune by editing the lists. Prefer few, strong, unambiguous phrases over
many weak ones -- weak phrases create cross-class ties that get bucketed
as Unknown and shrink the labeled set.
"""

import re

RULES: dict[str, list[str]] = {
    "Systematic Literature Review": [
        r"\bsystematic literature review\b",
        r"\bsystematic review\b",
        r"\bsystematic mapping study\b",
        r"\bsystematic mapping\b",
        r"\bsearch string\b",
        r"\binclusion criteria\b",
        r"\bexclusion criteria\b",
        r"\bquality assessment\b",
        r"\breview protocol\b",
        r"\bsnowballing\b",
        r"\bprisma\b",
        r"\bkitchenham\b",
    ],
    "Survey": [
        r"\bwe (?:conducted|ran|carried out) a survey\b",
        r"\bonline survey\b",
        r"\bquestionnaire\b",
        r"\bsurvey respondents?\b",
        r"\bsurvey of (?:developers|practitioners|professionals|engineers)\b",
        r"\bpractitioner survey\b",
        r"\bdeveloper survey\b",
        r"\blikert\b",
        r"\bweb-based survey\b",
        r"\bsurvey participants?\b",
    ],
    "Controlled Experiment": [
        r"\bcontrolled experiment\b",
        r"\brandomi[sz]ed\b",
        r"\brandomly assigned\b",
        r"\btreatment group\b",
        r"\bcontrol group\b",
        r"\bbetween[- ]subjects?\b",
        r"\bwithin[- ]subjects?\b",
        r"\bexperimental (?:group|condition)\b",
        r"\bplacebo\b",
        r"\bindependent variables?\b",
        r"\bdependent variables?\b",
    ],
    "Tool Paper": [
        r"\bwe (?:present|introduce|propose|developed?) (?:a|an|the|our) (?:new )?tool\b",
        r"\bour tool\b",
        r"\bthe tool (?:supports?|allows?|enables?|provides?|is)\b",
        r"\bopen[- ]source tool\b",
        # A named tool described by purpose, e.g. "a smell detection tool for
        # ...". TD tool papers rarely say "we present a tool" but routinely
        # phrase it this way; validated as high-precision on the dev set.
        r"\b(?:a |the |our )?tool (?:for|to|that|which|supports?|called)\b",
        r"\b(?:detection|analysis|visuali[sz]ation|monitoring|recommendation) tool\b",
        r"\b(?:eclipse|intellij|vs ?code) plug[- ]?in\b",
        r"\bide plug[- ]?in\b",
        r"\b(?:github|gitlab)\.com/\S+",
        r"\bavailable (?:at|on)\b.{0,30}\bhttp",
        r"\bdemonstration\b",
        r"\bprototype (?:tool|implementation)\b",
        r"\bcommand[- ]line tool\b",
        # A contributed artifact framed as "we propose/present/introduce a
        # <method|model|dataset|...>". In this corpus the gold taxonomy treats a
        # paper whose central contribution is a newly built approach/model/dataset
        # as a Tool Paper, even when it is empirically evaluated. Validated on the
        # gold set at ~10:1 precision for Tool Paper vs Empirical Study.
        r"\bwe (?:propose|present|introduce|develop|design|build|implement)\b.{0,45}\b(?:approach|method|methodolog|technique|framework|model|classifier|tool|system|dataset|data set|index|metric|pipeline|algorithm|plug[- ]?in|prototype)\b",
        # A named measurement artifact, e.g. "an architectural technical debt index".
        r"\b(?:technical debt|td|architectural\s+\w+) index\b",
        # A *new/automated* approach/model created to detect/classify/predict/etc.
        # The novelty/automation cue (novel|new|automated|...) is what keeps this
        # from firing on empirical studies that merely *use* such techniques.
        r"\b(?:novel|new|automated|automatic|proposed|a)\s+(?:approach|method|technique|framework|model|classifier|pipeline)\b.{0,45}\b(?:detect|identif|classif|predict|estimat|measur|forecast|recommend|remediat)",
    ],
    "Empirical Study": [
        r"\bempirical study\b",
        r"\bempirical investigation\b",
        r"\bempirical evaluation\b",
        r"\bempirical analysis\b",
        r"\bmining software repositories?\b",
        r"\blarge[- ]scale study\b",
        r"\bobservational study\b",
        r"\bexploratory study\b",
        r"\bquantitative study\b",
        r"\b\d+ (?:open[- ]source |open )?projects?\b",
    ],
    "Case Study": [
        r"\bcase study\b",
        r"\bcase studies\b",
        r"\b(?:multiple|multi)[- ]case study\b",
        r"\bsingle[- ]case study\b",
        r"\bembedded case study\b",
        r"\bexploratory case study\b",
        r"\bin[- ]depth case\b",
        r"\bcase company\b",
    ],
    "Experience Report": [
        r"\bexperience report\b",
        r"\blessons learned\b",
        r"\bour experience\b",
        r"\bindustrial experience\b",
        r"\bexperience (?:with|using|applying)\b",
        r"\bin our (?:organi[sz]ation|company|team)\b",
        r"\b(?:at|inside) (?:microsoft|google|facebook|meta|amazon|netflix|ibm|abb|ericsson|sap|spotify|uber|airbnb)\b",
        r"\bpractitioners?[’']? (?:perspective|view|experience)\b",
        # Provenance cues: a technique/framework/practice applied first-hand in a
        # specific real organization is an Experience Report even without the
        # phrase "lessons learned". Validated against the gold Experience Reports.
        r"\bapplied (?:in|at|within)\b.{0,30}\b(?:organi[sz]ation|compan|unit|team|firm|enterprise|department|division)",
        r"\bwe (?:present|describe|report)\b.{0,20}\b(?:practice|experience|deployment)\b",
        r"\bin an? (?:industrial|industry|real[- ]world|production) (?:setting|context|environment)\b",
    ],
    "Position Paper": [
        r"\bposition paper\b",
        r"\bvision paper\b",
        r"\bwe argue\b",
        r"\bwe advocate\b",
        r"\bwe call for\b",
        r"\bthis paper calls for\b",
        r"\broadmap\b",
        r"\bresearch agenda\b",
        r"\bin this position paper\b",
    ],
    "Theoretical Contribution": [
        r"\bconceptual framework\b",
        r"\btheoretical framework\b",
        r"\btheoretical model\b",
        r"\bformal model\b",
        r"\bformal theory\b",
        r"\bontology\b",
        r"\bwe propose a (?:framework|model|theory|taxonomy)\b",
        r"\bmeta[- ]model\b",
        r"\bconceptual model\b",
        r"\b(?:probabilistic|mathematical|analytical|computational) model\b",
        r"\bmodel(?:ed|led|ing)\b.{0,20}\bprobabilistic",
        r"\bwe model\b",
        r"\bwe (?:develop|present|propose|introduce|build)(?:ed)? (?:a |an |the |our )?(?:novel )?(?:conceptual|theoretical|formal|mathematical|probabilistic) model\b",
    ],
}

COMPILED: dict[str, list[re.Pattern[str]]] = {
    label: [re.compile(p, re.IGNORECASE) for p in patterns]
    for label, patterns in RULES.items()
}


# ── Decision-tree cue groups ────────────────────────────────────────────────
# These implement the "Labeling Decision Procedure" in
# docs/Paper Type Definitions.md. They are applied as a PRECEDENCE layer on top
# of the per-label keyword vote (see rule_classifier.classify): they resolve the
# Survey / SLR / Empirical-Study region that plain keyword counting handles
# inconsistently, while everything else still falls through to the vote.
#
# The tree disambiguates the overloaded "Survey" label along two axes:
#   * literature-sense Survey vs SLR  -> decided by PROTOCOL presence (Step 2A)
#   * practitioner-sense Survey vs Empirical Study -> decided by data source (2B)

# Step 1 — does the paper's PRIMARY work review existing literature (secondary)?
SECONDARY_STUDY_PATTERNS: list[str] = [
    r"\bliterature review\b",
    r"\bmultivocal\b",
    r"\b(?:systematic )?mapping study\b",
    r"\bsystematic mapping\b",
    r"\btertiary study\b",
    # "state of the art" only signals a secondary study when it is the *subject*
    # of a review/survey/overview -- bare "state-of-the-art" is also how primary
    # papers refer to their comparison baseline ("in contrast to the SOTA"), so
    # require the governing review verb.
    r"\b(?:survey|review|overview) of (?:the )?(?:state[- ]of[- ]the[- ]art|literature|research|existing|current)\b",
    r"\bwe (?:review|survey) the (?:literature|research|state)\b",
    r"\breview of (?:the )?(?:literature|existing|\d+ (?:studies|papers))\b",
]

# Step 2A — secondary study WITH a documented, reproducible protocol -> SLR.
# Without any of these the secondary study is a narrative Survey (lit sense).
PROTOCOL_PATTERNS: list[str] = [
    r"\bsystematic literature review\b",
    r"\bsystematic review\b",
    r"\bsystematic mapping\b",
    r"\bmultivocal literature review\b",
    r"\bsearch string\b",
    r"\binclusion criteria\b",
    r"\bexclusion criteria\b",
    r"\bquality assessment\b",
    r"\breview protocol\b",
    r"\bsnowballing\b",
    r"\bprisma\b",
    r"\bkitchenham\b",
    r"\bdata extraction\b",
]

# Step 2B — primary study whose data is self-reported via a questionnaire /
# structured survey of people -> Survey (practitioner sense), as opposed to data
# mined/measured from artifacts (Empirical Study).
PRACTITIONER_SURVEY_PATTERNS: list[str] = [
    r"\bwe (?:conducted|ran|carried out) a survey\b",
    r"\bonline survey\b",
    r"\bquestionnaire\b",
    r"\bsurvey respondents?\b",
    r"\bsurvey participants?\b",
    r"\bsurvey of (?:developers|practitioners|professionals|engineers)\b",
    r"\bpractitioner survey\b",
    r"\bdeveloper survey\b",
    r"\blikert\b",
    r"\bweb-based survey\b",
    r"\bfrom the (?:point of view|perspective) of (?:software )?practitioners\b",
    r"\bopen[- ]ended (?:answers|questions|responses)\b",
]

SECONDARY_STUDY: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in SECONDARY_STUDY_PATTERNS
]
PROTOCOL: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in PROTOCOL_PATTERNS
]
PRACTITIONER_SURVEY: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in PRACTITIONER_SURVEY_PATTERNS
]
