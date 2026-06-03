### Paper Type Taxonomy Definitions

## Labeling Decision Procedure

Many papers superficially fit more than one type. In particular **Survey** is
overloaded across two axes — a *literature* survey is a secondary study that
looks like an **SLR**, while a *practitioner* survey is a primary study that
looks like an **Empirical Study**. Resolve every secondary/empirical paper with
the decision tree below before falling back to the per-type definitions.

```
Step 1 — What is the paper's PRIMARY data?
  (a) existing published literature      -> SECONDARY study -> go to Step 2A
  (b) newly collected data               -> PRIMARY study   -> go to Step 2B

Step 2A — Secondary study: SLR vs Survey (literature sense)
  - Documented, reproducible protocol
    (research questions + search string + inclusion/exclusion criteria,
     usually quality assessment / data extraction)         -> Systematic Literature Review
    * Systematic mapping studies and multivocal literature
      reviews (MLRs) WITH a protocol count as SLR.
  - Narrative / curated review, "overview", "state of the art",
    no reproducible selection process                       -> Survey (literature sense)

Step 2B — Primary study: Survey vs Empirical Study vs others
  - Data is self-reported by people via questionnaire /
    structured survey (opinions, experiences, perceptions)  -> Survey (practitioner sense)
  - Data is mined or measured from software artifacts
    (repositories, code, defects, projects)                 -> Empirical Study
  - Deliberate manipulation of variables w/ control group   -> Controlled Experiment
  - In-depth single/few cases in real context, multi-source -> Case Study
  - First-person account in one org, lessons learned        -> Experience Report
```

**Mixed-method tiebreak.** If a primary study uses BOTH a practitioner
questionnaire AND artifact analysis, label it by the method that answers the
paper's primary research questions / produces its headline findings (dominant
contribution).

### Cue words for the two hard edges

| Signal | Points to |
|---|---|
| "systematic literature review", "systematic mapping study", "multivocal literature review", "search string", "inclusion/exclusion criteria", "quality assessment", "PRISMA", "snowballing", "review protocol" | **SLR** |
| "overview of", "state of the art", "we survey the literature", "narrative review" — **and none** of the SLR protocol cues above | **Survey (literature)** |
| "we conducted a survey", "questionnaire", "respondents", "from the point of view of practitioners", "open-ended answers", Likert scale | **Survey (practitioner)** |
| "we mined", "repository/repositories", "we analyzed N projects", "dataset of", measured metrics | **Empirical Study** |

---

### Per-Type Definitions

**Empirical Study**
An empirical study collects and analyzes real-world data to answer a research question, meaning that the researcher does not control the conditions. The goal for empirical studies is observation rather than intervention.
*vs Survey (practitioner sense): the discriminator is the data source — mined/measured artifacts here vs. self-reported questionnaire data for Survey (see Step 2B).*

**Controlled Experiment**
A controlled experiment involves deliberately manipulating one of more variables to measure the effects, having a control group for comparison. Participants are typically assigned to conditions and results are analysed statistically.

**Systematic Literature Review (SLR)**
A systemic literature review follows a formally defined, reproducible protocol to identify, select, and synthesise all existing research on a topic. It includes a documented search strategy, inclusion/exclusion criteria, and quality assessment. Another researcher following the same protocol should get the same result.
*vs Survey: the discriminator is a documented, reproducible protocol (see Step 2A). Systematic mapping studies and multivocal literature reviews (MLRs) with a protocol are SLRs.*

**Survey**
"Survey" covers two distinct sub-senses; use the decision tree above to pick the right one.
- **Survey (literature sense)** — a broad, narrative review of existing work *without* a reproducible protocol. It is a secondary study like an SLR, but distinguished from one purely by the *absence* of a documented, reproducible search/selection process (the Step 2A test). Selection is curated by the authors; it is less reproducible than an SLR but often broader in scope. Typical signals: "overview", "state of the art".
- **Survey (practitioner sense)** — a primary study whose main data is collected by questionnaire or structured survey of practitioners' opinions and experiences. It is distinguished from an Empirical Study by its data source being *self-report* rather than mined/measured artifacts (the Step 2B test). Typical signals: "we conducted a survey", "questionnaire", "respondents".

**Tool Paper**
A tool paper mainly focuses on software/hardware tools, describing what the tool does and how it can be used. The paper then evaluates its effectiveness and correctness.

**Experience Report**
An experience report is an authentic, detailed account of a technique, process, or tool applied in a real organizational or project context. The findings are not intended to be universally generalisable and may include qualitative data (interviews, observations, document analysis) and quantitative metrics.

**Case Study**
A case study is an in-depth, detailed investigation of a single case (or a small number of cases), such as an individual, team, organization, project, or event, examined within its real-world context. It typically uses multiple sources of evidence (e.g., interviews, observations, documents, and metrics) to understand how and why phenomena occur, often with an emphasis on context-specific insights rather than broad generalization.

**Position Paper**
A position paper is more opinion-driven, arguing for a particular viewpoint, agenda, or direction for future research, without presenting new empirical data or a new tool. It is intended to provoke discussion in the community.

**Theoretical Contribution**
A theoretical contribution proposes a new model, framework, taxonomy, or formal theory to explain or structure a phenomenon, without necessarily validating it empirically. The contribution advances how we think about or formalise a problem rather than solving a specific instance of it.
