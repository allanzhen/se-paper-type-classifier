# Limitations

Plain-language summary of what could go wrong with our data, classifier, and analysis — and what we did about it.

---

## Data Limitations

**1. We only used top-tier venues (CORE A and A*)**
We restricted the corpus to high-quality venues to control for paper quality. The downside is that Experience Reports and Position Papers appear more often in workshops and lower-tier venues, so our corpus likely undercounts those types. The 4% Experience Report figure is probably lower than the true proportion across all TD research.

**Mitigation**: We acknowledge this scope limitation explicitly in the paper and treat findings about rare types with caution.

---

**2. We only have 342 papers**
After filtering to A/A* venues and 2010 onwards, 683 papers matched. Of those, only 342 had abstracts available on Semantic Scholar. The other 341 were dropped because our classifier needs an abstract to work.

**Mitigation**: None — this is a hard constraint of the pipeline. The 342 papers are still a meaningful and consistently-sourced corpus.

---

**3. The corpus is sparse before 2018**
Only a handful of papers exist in the corpus before 2018 (e.g. 1 paper in 2010, 1 in 2013). This makes temporal comparisons between early and recent years unreliable.

**Mitigation**: We grouped papers into two broader eras (2015–2019 vs 2020–2026) rather than comparing year by year, to avoid drawing conclusions from single-digit sample sizes.

---

## Analysis Limitations

**4. The classifier only reads titles and abstracts**
Paper type is not always obvious from the abstract alone. A tool paper that was evaluated empirically can look like an empirical study. A survey with no questionnaire language can look like an empirical study. We found that roughly 20% of our development set could not be reliably classified from the abstract alone.

**Mitigation**: We built a 100-paper gold standard by manually reading each paper and labelling it. We report honest accuracy metrics rather than assuming the classifier is correct.

---

**5. The classifier has a bias toward Empirical Study**
The zero-shot NLI component tends to default to Empirical Study for ambiguous papers because most Technical Debt abstracts contain measurement and analysis language regardless of paper type. 5 of the 20 classifier errors resulted in a paper being wrongly assigned to Empirical Study.

**Mitigation**: The rule-based component (Stage 1) uses explicit keyword patterns to catch classes with unambiguous vocabulary (SLR, Survey, Controlled Experiment) before the zero-shot model runs. This reduces but does not eliminate the bias.

---

**6. Rare classes have too few gold standard examples**
The gold standard has only 3 Position Papers, 5 Surveys, 5 Experience Reports, and 5 Theoretical Contributions. With so few examples, the classifier cannot reliably learn those classes and the F1 scores for them are noisy estimates.

**Mitigation**: We flag these classes explicitly — Position Paper (F1=0.50) and Survey (F1=0.57) should be treated as lower bounds, not stable measurements. A gold standard of 40–50 examples per class would be needed for reliable estimates.

---

**7. Some gold standard labels were revised after initial labelling**
Five papers were reclassified after initial labelling, all moving from Survey to either SLR or Empirical Study. The Survey/SLR boundary is genuinely ambiguous — not all literature reviews explicitly state whether they followed a formal systematic protocol.

**Mitigation**: All reclassifications were documented (see the git history of `data/gold/gold_standard_papers.csv`) and the final labels reflect the agreed definitions in our taxonomy.

---

**8. The chi-square test for temporal trends has low power**
The pre-2020 period only has 72 papers. With such a small group, the chi-square test may fail to detect real shifts simply because the sample is too small — not because no shift occurred.

**Mitigation**: We report the result as "no statistically significant shift" rather than "no shift", and separately note the directional trend in SLRs (1.4% to 5.2%) as a descriptive observation.
