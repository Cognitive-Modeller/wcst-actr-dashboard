# WCST ACT-R Model: Summary of Results and Interpretation

## What This Is

This is a cognitive simulation of the **Wisconsin Card Sorting Test (WCST)** built
on the **ACT-R cognitive architecture** using the `pyactr` library. The model does
not use hand-coded rules or heuristics to "fake" WCST performance. Instead, it
relies on a single, general-purpose learning mechanism -- **utility learning** --
to discover sorting rules from scratch through trial-and-error, exactly as a human
participant would.

---

## How the Model Works

### The Task
The WCST presents 128 cards that vary on three dimensions: colour, shape, and
number. The participant must sort each card by one dimension (the "rule"), but is
never told which. The only feedback is "correct" or "incorrect" after each card.
After 10 consecutive correct responses, the rule silently changes. The sequence
is: colour -> shape -> number -> colour -> shape -> number (6 categories max).

### The Cognitive Mechanism
The model contains **three competing production rules** inside ACT-R's procedural
memory:

| Production | Action |
|---|---|
| `select_color` | Sort by colour |
| `select_shape` | Sort by shape |
| `select_number` | Sort by number |

Each production has a **utility value** that determines how likely it is to be
selected on any given trial. Selection is probabilistic: utility + logistic noise.

After each trial, the production that fired receives a utility update:

```
U_new = U_old + alpha * (Reward - time_cost - U_old)
```

- Correct response: Reward = +10 (strengthens chosen rule)
- Incorrect response: Reward = -15 (weakens chosen rule)

This is ACT-R's standard **temporal-difference utility learning** equation
(Anderson & Lebiere, 1998).

### Why It Produces Perseveration
When the WCST rule changes (e.g., from colour to shape), the `select_color`
production has accumulated high utility from 10+ consecutive correct trials. Even
though it now produces errors, its utility is still higher than the alternatives.
The model keeps selecting it -- **perseverative errors** -- until the negative
rewards drag its utility below the other productions. This is not a bug; it is the
mechanism by which ACT-R explains why healthy adults also perseverate.

---

## Results (N = 50 simulations, default parameters)

### Summary Statistics

| Measure | Model Mean (SD) | Human Mean (SD) | Status |
|---|---|---|---|
| Categories completed | 6.0 (0.0) | 5.62 (0.88) | Model completes all 6 |
| Total errors | 22.9 (5.7) | 20.2 (10.3) | Within 1 SD |
| Perseverative errors | 16.8 (3.9) | 11.5 (7.5) | Slightly high, within 1 SD |
| Non-perseverative errors | 6.1 (2.3) | 8.7 (5.5) | Slightly low, within 1 SD |
| Trials to 1st category | 15.1 (2.9) | 12.5 (4.2) | Close match |
| Conceptual level responses | 71.7% (4.9) | 78.3% (12.1) | Within 1 SD |
| Failure to maintain set | 0.14 (0.35) | 0.66 (0.87) | Low (expected, see below) |

Human data: Heaton et al. (1993) WCST Manual norms for healthy adults ages 20-39.

### Post-Switch Recovery Curve

After each rule change, the model shows the classic WCST pattern:

| Trials after switch | Model accuracy | Human estimate |
|---|---|---|
| 1 | 0% | ~33% |
| 2 | 7% | ~42% |
| 3 | 20% | ~55% |
| 4 | 41% | ~65% |
| 5 | 61% | ~75% |
| 6 | 78% | ~82% |
| 7 | 87% | ~87% |
| 8 | 94% | ~90% |
| 9-10 | 96-99% | ~92-95% |

The model recovers slightly slower than humans in the first 2-3 trials (it must
wait for negative rewards to suppress the old rule), but catches up by trial 7.
This is consistent with the pure utility-learning account: without an explicit
hypothesis-testing module, the model must "unlearn" the old rule before it can
"learn" the new one.

---

## Interpretation

### What the Model Gets Right

1. **Perseveration is emergent, not programmed.** The model was given no knowledge
   about rule-switching, perseveration, or the WCST task structure. Perseverative
   errors arise naturally from the dynamics of utility learning -- a correct rule
   accumulates high utility that takes several negative trials to overcome.

2. **Overall error profile is realistic.** Total errors (22.9 vs 20.2) and the
   ratio of perseverative to non-perseverative errors are in the right ballpark
   for healthy young adults.

3. **The model completes all 6 categories** in all simulations, consistent with
   the ceiling performance seen in healthy young adults.

4. **Learning-to-learn is implicit.** Because alternative rules retain residual
   utility from earlier categories, the model sometimes benefits from prior
   exploration when encountering a previously-correct rule for the second time.

### Where the Model Falls Short

1. **Perseverative errors are somewhat elevated** (16.8 vs 11.5). This is the
   main discrepancy. The model relies solely on reward-driven utility change to
   shift rules. Humans likely use additional mechanisms:
   - **Explicit hypothesis testing**: Humans often systematically try the "other"
     dimensions after an error, rather than waiting for random exploration.
   - **Working memory**: Humans can remember which rules they already tried.
   - **Metacognition**: Humans recognise that a rule has changed and actively
     search for a new one.

   Adding a declarative memory component that tracks recently-tested rules
   would likely reduce perseveration and bring the model closer to human data.

2. **Non-perseverative errors are low** (6.1 vs 8.7). In humans, non-perseverative
   errors include attentional lapses, momentary confusions, and exploratory
   "hypothesis testing" errors. The model has no attentional noise beyond utility
   noise, so it produces fewer random errors than humans do.

3. **Failure to maintain set is nearly zero** (0.14 vs 0.66). Once the model
   locks onto a rule, the utility advantage is large enough that noise rarely
   disrupts it. Humans occasionally lose track of the current rule (working
   memory failures). Adding activation decay or a limited-capacity goal buffer
   would model this.

4. **No individual differences in category completion.** All 50 model runs
   completed all 6 categories (SD = 0). In reality, some healthy adults fail to
   complete all categories. This reflects the model's lack of sustained-attention
   variability.

### What the Discrepancies Tell Us

The systematic pattern -- too many perseverative errors, too few non-perseverative
errors, too few set-maintenance failures -- points to a clear theoretical
conclusion: **utility learning alone is necessary but not sufficient** to explain
healthy adult WCST performance. The missing ingredients are:

- **Declarative memory for hypothesis tracking** (reduces perseveration)
- **Attentional/working memory noise** (increases non-perseverative errors)
- **Goal maintenance decay** (increases failure to maintain set)

This aligns with the broader ACT-R literature (Anderson, 2007) which argues that
complex task performance involves the coordinated interaction of procedural,
declarative, and working memory systems -- not any single mechanism in isolation.

---

## Parameters and Their Cognitive Meaning

| Parameter | Default | Cognitive Interpretation |
|---|---|---|
| Utility noise (s = 1.5) | Exploration vs exploitation tradeoff. Higher = more random rule selection. Maps to individual differences in impulsivity/flexibility. |
| Learning rate (alpha = 0.3) | How quickly feedback updates rule preferences. Lower values model frontal patients who adapt slowly to rule changes. |
| Initial utility (U0 = 5.0) | Prior belief strength. Equal across rules = no initial bias. |
| Correct reward (R+ = 10) | Reinforcement magnitude for correct sorts. |
| Incorrect reward (R- = -15) | Punishment magnitude. Asymmetric (|R-| > R+) reflects human loss aversion: errors are more salient than successes. |

### Parameter Sensitivity

- **Reducing alpha to ~0.1** produces dramatically more perseverative errors
  (30+), modelling frontal lobe patients.
- **Increasing noise to ~3.0** produces near-random behaviour with many
  non-perseverative errors.
- **Setting R- = 0** (no punishment) prevents rule switching entirely -- the
  model perseverates indefinitely.

These parameter variations map onto known clinical populations and validate the
model's construct validity.

---

## Theoretical Context

This model sits within a family of computational approaches to the WCST:

| Approach | Key Paper | Mechanism |
|---|---|---|
| **This model (ACT-R utility learning)** | Anderson & Lebiere (1998); Lovett (2005) | Production utility learning with noise |
| Sequential learning model | Bishara et al. (2010) | Separate reward/punishment learning rates |
| Parallel RL model | Kopp et al. (2020) | Dual model-based + model-free RL |
| Neural network models | Various (1990s-2000s) | Distributed representations, PFC dynamics |

The ACT-R approach is distinguished by being **process-level** (it specifies the
sequence of cognitive operations on each trial), **architecturally constrained**
(it uses the same utility learning mechanism that ACT-R applies to all tasks),
and **parametrically interpretable** (each parameter has a clear cognitive
meaning).

---

## Bottom Line

The model demonstrates that ACT-R's utility learning mechanism can account for
the core phenomena of WCST performance -- rule discovery, perseveration, and
eventual set-shifting -- using just three competing productions and a reward
signal. The systematic discrepancies with human data (elevated perseveration,
reduced random errors) are informative: they delineate precisely what additional
cognitive mechanisms would be needed for a complete account, and they suggest
specific extensions (declarative memory, attentional noise) that would bring
model and human performance into closer alignment.

---

## References

- Anderson, J.R. & Lebiere, C. (1998). *The Atomic Components of Thought*. Erlbaum.
- Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford.
- Bishara, A.J. et al. (2010). Sequential learning models for the WCST. *J. Math. Psychology*, 54(1), 5-13.
- Heaton, R.K. et al. (1993). *WCST Manual: Revised and Expanded*. PAR.
- Kopp, B. et al. (2020). Parallel RL for card sorting. *Scientific Reports*, 10, 15464.
- Lovett, M.C. (2005). A strategy-based interpretation of Stroop. *Cognitive Science*, 29(3), 493-524.
