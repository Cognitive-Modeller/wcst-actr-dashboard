# WCST ACT-R Simulation Dashboard

An interactive web dashboard for exploring a cognitive simulation of the **Wisconsin Card Sorting Test (WCST)** built on the **ACT-R cognitive architecture** using [pyactr](https://github.com/jakdot/pyactr).

The model does not use hand-coded rules or heuristics. Instead, it relies on a single, general-purpose learning mechanism -- **utility learning** -- to discover sorting rules from scratch through trial-and-error, exactly as a human participant would.

## Demo

https://github.com/Cognitive-Modeller/wcst-actr-dashboard/releases/download/v0.1.0/demo.webm

## How it works

### The task

The WCST presents 128 cards varying on three dimensions: colour, shape, and number. The participant sorts each card by one dimension (the "rule") but is never told which. The only feedback is "correct" or "incorrect". After 10 consecutive correct responses the rule silently changes, cycling through: colour -> shape -> number -> colour -> shape -> number (6 categories max).

### The cognitive mechanism

Three competing **production rules** in ACT-R's procedural memory (`select_color`, `select_shape`, `select_number`) each carry a **utility value** determining selection probability. After each trial the fired production receives a temporal-difference utility update:

```
U_new = U_old + alpha * (Reward - U_old)
```

When the rule changes, the old production still has high accumulated utility, producing **perseverative errors** until negative rewards drag it below the alternatives. This is not a bug -- it is the mechanism by which ACT-R explains why healthy adults also perseverate.

## Model variants

The dashboard implements **7 model variants** ranked by RMSD fit to human norms (Heaton et al., 1993):

| Variant | Mechanisms | Reference |
|---|---|---|
| **Basic** | Pure utility learning | Anderson & Lebiere (1998) |
| **Enhanced** | + DM hypothesis tracking, attentional lapses, set decay | Barcelo & Knight (2002) |
| **+ Asymmetric LR** | Punishment updates amplified 1.8x | Bishara et al. (2010) |
| **+ Lose-Shift** | Non-chosen rules boosted after errors | Nyhus & Barcelo (2009) |
| **+ Frustration** | Noise increases after 3+ consecutive errors | Daw et al. (2006) |
| **+ Utility Decay** | All utilities decay 1.5%/trial toward baseline | Altmann & Gray (2008) |
| **Combined** | All mechanisms with attenuated parameters | Anderson (2007) |

## Dashboard features

- **Quick comparison** -- Basic vs Enhanced model side-by-side with summary statistics, trial-by-trial accuracy, utility traces, and post-switch recovery curves
- **Full variant comparison** -- All 7 variants ranked by normalised RMSD against human data
- **Teaching mode** -- 3-model progression (Basic -> Enhanced -> Full) with narrative explanations
- **Bayesian optimisation** -- Optuna TPE sampler with Common Random Numbers (CRN) and a composite objective (summary stats + post-switch curve + trial accuracy) to find optimal parameters
- **Interactive parameter tuning** -- Adjust all 9 cognitive parameters via sliders and re-run in real time

## Quickstart

Requires Python 3.11+.

```bash
# Clone
git clone https://github.com/Cognitive-Modeller/wcst-actr-dashboard.git
cd wcst-actr-dashboard

# Install dependencies (using uv)
uv sync

# Run
uv run python app.py
```

Then open [http://localhost:5050](http://localhost:5050).

### Without uv

```bash
pip install flask optuna pyactr
python app.py
```

## Parameters

| Parameter | Default | Cognitive interpretation |
|---|---|---|
| Utility noise (*s*) | 1.5 | Exploration vs exploitation. Higher = more random rule selection |
| Learning rate (*alpha*) | 0.3 | How quickly feedback updates preferences. Lower = frontal-patient profile |
| Initial utility (*U0*) | 5.0 | Prior belief strength. Equal across rules = no initial bias |
| Correct reward (*R+*) | 10.0 | Reinforcement magnitude for correct sorts |
| Incorrect reward (*R-*) | -15.0 | Asymmetric punishment reflecting human loss aversion |
| Lapse rate | 0.04 | Probability of attentional lapse (random selection) per trial |
| Set-loss rate | 0.02 | Probability of losing set after 5+ consecutive correct |
| Set-loss strength | 0.5 | How much utility regresses toward baseline on set loss |
| Hypothesis boost | 4.0 | Utility boost for untried rules after DM recall of a failure |

## Project structure

```
.
 app.py              # Flask web server (5 API endpoints)
 wcst_model.py       # ACT-R model (basic, enhanced, 7 variants, Bayesian opt)
 templates/
    index.html      # Single-page dashboard (Bootstrap + Chart.js)
 pyproject.toml
 SUMMARY.md          # Detailed results and theoretical interpretation
```

## References

- Anderson, J.R. & Lebiere, C. (1998). *The Atomic Components of Thought*. Erlbaum.
- Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford.
- Altmann, E.M. & Gray, W.D. (2008). An integrated model of cognitive control in task switching. *Psychological Review*, 115(3), 602-639.
- Barcelo, F. & Knight, R.T. (2002). Both random and perseverative errors underlie WCST deficits in prefrontal patients. *Neuropsychologia*, 40(3), 349-356.
- Bishara, A.J. et al. (2010). Sequential learning models for the WCST. *J. Math. Psychology*, 54(1), 5-13.
- Daw, N.D. et al. (2006). Cortical substrates for exploratory decisions in humans. *Nature*, 441, 876-879.
- Heaton, R.K. et al. (1993). *WCST Manual: Revised and Expanded*. PAR.
- Nyhus, E. & Barcelo, F. (2009). The Wisconsin Card Sorting Test and the cognitive assessment of prefrontal executive functions. *Brain and Cognition*, 71(3), 437-451.

## License

MIT
