"""
Wisconsin Card Sorting Test (WCST) - ACT-R Cognitive Model
==========================================================

Two model variants:

BASIC MODEL (utility learning only)
  - 3 competing rule-selection productions + 2 feedback productions
  - Perseveration emerges from accumulated utility on old rule
  - No hypothesis tracking, no attentional noise, no goal decay

ENHANCED MODEL (full cognitive mechanisms)
  Adds three mechanisms identified as missing from the basic model:
  1. Hypothesis tracking (DM-driven): After errors, the model recalls
     which rules were already tried and boosts untried alternatives.
     Implemented via pyactr declarative memory retrieval that biases
     subsequent production utility.
  2. Attentional lapses: With probability p_lapse, the production
     system is bypassed and a random rule is selected, modelling
     momentary disengagement (Barcelo & Knight, 2002).
  3. Set maintenance decay: After sustained correct responding, there
     is a probability that the goal-buffer representation degrades,
     modelling the limited-capacity maintenance of task set in WM
     (Altmann & Gray, 2008).

References:
  Anderson & Lebiere (1998). The Atomic Components of Thought.
  Lovett (2005). Cognitive Science, 29(3), 493-524.
  Barcelo & Knight (2002). Neuropsychologia, 40(3), 349-356.
  Altmann & Gray (2008). Psychological Review, 115(3), 602-639.
  Heaton et al. (1993). WCST Manual: Revised and Expanded. PAR.
"""

import pyactr as actr
import numpy as np
import warnings
import time as time_module
from collections import defaultdict

warnings.filterwarnings("ignore")

# ============================================================================
# HUMAN BENCHMARK DATA (Healthy adults, ages 20-39)
# ============================================================================
HUMAN_DATA = {
    "categories_completed":        {"mean": 5.62, "sd": 0.88,  "label": "Categories Completed"},
    "total_errors":                {"mean": 20.2, "sd": 10.3,  "label": "Total Errors"},
    "perseverative_errors":        {"mean": 11.5, "sd": 7.5,   "label": "Perseverative Errors"},
    "perseverative_responses":     {"mean": 12.8, "sd": 8.5,   "label": "Perseverative Responses"},
    "non_perseverative_errors":    {"mean": 8.7,  "sd": 5.5,   "label": "Non-Perseverative Errors"},
    "trials_first_category":      {"mean": 12.5, "sd": 4.2,   "label": "Trials to 1st Category"},
    "failure_to_maintain_set":    {"mean": 0.66, "sd": 0.87,  "label": "Failure to Maintain Set"},
    "conceptual_level_responses":  {"mean": 78.3, "sd": 12.1,  "label": "Conceptual Level Resp. (%)"},
    "total_trials_used":           {"mean": 100.5,"sd": 18.7,  "label": "Total Trials Used"},
    "percent_errors":              {"mean": 21.5, "sd": 9.8,   "label": "Percent Errors (%)"},
    "percent_perseverative_errors":{"mean": 11.2, "sd": 6.8,   "label": "Percent Persev. Errors (%)"},
}

# WCST constants
RULE_SEQUENCE = ["color", "shape", "number", "color", "shape", "number"]
CRITERION = 10
MAX_TRIALS = 128
MAX_CATEGORIES = 6
RULES = ["color", "shape", "number"]

_CHUNK_TYPES_DEFINED = False
_ENHANCED_CHUNK_TYPES_DEFINED = False


def _ensure_chunk_types():
    global _CHUNK_TYPES_DEFINED
    if not _CHUNK_TYPES_DEFINED:
        actr.chunktype("wcst_goal", "state, chosen, correct_rule")
        _CHUNK_TYPES_DEFINED = True


def _ensure_enhanced_chunk_types():
    global _ENHANCED_CHUNK_TYPES_DEFINED
    _ensure_chunk_types()
    if not _ENHANCED_CHUNK_TYPES_DEFINED:
        actr.chunktype("rule_memory", "rule, outcome")
        actr.chunktype("recall_goal", "state, retrieved_rule")
        _ENHANCED_CHUNK_TYPES_DEFINED = True


# ============================================================================
# BASIC MODEL
# ============================================================================

def _create_trial_model(utilities, correct_rule, params):
    """Create a 5-production ACT-R model for one WCST trial."""
    _ensure_chunk_types()
    model = actr.ACTRModel(
        subsymbolic=True,
        utility_noise=params["utility_noise"],
        utility_learning=True,
        utility_alpha=params["utility_alpha"],
        rule_firing=0.05,
    )
    model.goal.add(actr.chunkstring(string=f"""
        isa     wcst_goal
        state   select
        chosen  None
        correct_rule {correct_rule}
    """))
    for rule in RULES:
        model.productionstring(
            name=f"select_{rule}",
            string=f"""
            =g>
            isa     wcst_goal
            state   select
            ==>
            =g>
            isa     wcst_goal
            state   check
            chosen  {rule}
            """,
            utility=utilities.get(f"select_{rule}", params["initial_utility"]),
        )
    model.productionstring(
        name="fb_correct",
        string="""
        =g>
        isa     wcst_goal
        state   check
        chosen  =x
        correct_rule =x
        ==>
        =g>
        isa     wcst_goal
        state   done
        """,
        utility=utilities.get("fb_correct", params["initial_utility"]),
        reward=params["correct_reward"],
    )
    model.productionstring(
        name="fb_incorrect",
        string="""
        =g>
        isa     wcst_goal
        state   check
        chosen  =x
        correct_rule ~=x
        ==>
        =g>
        isa     wcst_goal
        state   done
        """,
        utility=utilities.get("fb_incorrect", params["initial_utility"]),
        reward=params["incorrect_reward"],
    )
    return model


def _run_single_trial(utilities, correct_rule, params):
    """Execute one basic WCST trial. Returns (chosen, correct, new_utilities)."""
    model = _create_trial_model(utilities, correct_rule, params)
    sim = model.simulation(trace=False)
    sim.run(max_time=2.0)

    new_utilities = {}
    for name in ["select_color", "select_shape", "select_number",
                  "fb_correct", "fb_incorrect"]:
        new_utilities[name] = model.productions[name]["utility"]

    chosen_rule = None
    for rule in RULES:
        key = f"select_{rule}"
        old_u = utilities.get(key, params["initial_utility"])
        new_u = new_utilities[key]
        if abs(new_u - old_u) > 1e-6:
            chosen_rule = rule
            break

    if chosen_rule is None:
        best = max(RULES, key=lambda r: new_utilities[f"select_{r}"])
        chosen_rule = best

    is_correct = (chosen_rule == correct_rule)
    return chosen_rule, is_correct, new_utilities


# ============================================================================
# ENHANCED MODEL - three additional cognitive mechanisms
# ============================================================================

def _run_dm_retrieval(failed_rules, params):
    """
    Use pyactr declarative memory to attempt retrieval of a previously
    failed rule.  Returns the retrieved rule name or None.

    The model populates DM with chunks for each failed rule, then a
    production requests retrieval.  Activation-based competition means
    the most recent / frequent failure is most likely retrieved.
    """
    if not failed_rules:
        return None

    import io, sys

    _ensure_enhanced_chunk_types()

    model = actr.ACTRModel(
        subsymbolic=True,
        retrieval_threshold=-2.0,
        decay=0.5,
        rule_firing=0.05,
    )

    dm = model.decmem
    base_time = 0.0
    for i, rule in enumerate(failed_rules):
        chunk = actr.makechunk(
            nameofchunk=f"fail_{rule}_{i}",
            typename="rule_memory",
            rule=rule,
            outcome="fail",
        )
        dm.add(chunk, time=base_time + i * 0.1)

    model.goal.add(actr.chunkstring(string="""
        isa     recall_goal
        state   start
        retrieved_rule None
    """))

    model.productionstring(name="attempt_recall", string="""
        =g>
        isa     recall_goal
        state   start
        ==>
        +retrieval>
        isa     rule_memory
        outcome fail
        =g>
        isa     recall_goal
        state   waiting
    """)

    for rule in RULES:
        model.productionstring(name=f"recalled_{rule}", string=f"""
            =g>
            isa     recall_goal
            state   waiting
            =retrieval>
            isa     rule_memory
            rule    {rule}
            ==>
            =g>
            isa     recall_goal
            state   done
            retrieved_rule {rule}
        """)

    model.productionstring(name="recall_failed", string="""
        =g>
        isa     recall_goal
        state   waiting
        ?retrieval>
        state   error
        ==>
        =g>
        isa     recall_goal
        state   done
        retrieved_rule None
    """)

    # Run with trace to detect which recalled_X production fired
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        sim = model.simulation(trace=True)
        sim.run(max_time=1.0)
        trace = buffer.getvalue()
    finally:
        sys.stdout = old_stdout

    for rule in RULES:
        if f"recalled_{rule}" in trace:
            return rule

    return None


def _run_enhanced_single_trial(utilities, correct_rule, params, trial_state):
    """
    Execute one enhanced WCST trial with all three cognitive mechanisms.

    Parameters
    ----------
    utilities : dict
        Current production utilities.
    correct_rule : str
        Active sorting rule.
    params : dict
        Full parameter set (basic + enhanced).
    trial_state : dict
        Mutable state tracking hypothesis history, consecutive correct, etc.

    Returns
    -------
    tuple: (chosen_rule, is_correct, updated_utilities, updated_trial_state)
    """
    lapse_rate = params.get("lapse_rate", 0.04)
    set_loss_rate = params.get("set_loss_rate", 0.06)
    set_loss_strength = params.get("set_loss_strength", 0.5)
    hypothesis_boost = params.get("hypothesis_boost", 4.0)
    init_u = params["initial_utility"]

    # ------------------------------------------------------------------
    # 1. ATTENTIONAL LAPSE
    #    With probability lapse_rate the production system is bypassed.
    #    Models momentary disengagement (Barcelo & Knight, 2002).
    # ------------------------------------------------------------------
    if np.random.random() < lapse_rate:
        chosen = np.random.choice(RULES)
        is_correct = (chosen == correct_rule)
        trial_state["lapse"] = True
        return chosen, is_correct, utilities, trial_state

    trial_state["lapse"] = False

    # ------------------------------------------------------------------
    # 2. SET MAINTENANCE DECAY
    #    After >=5 consecutive correct, there is a probability that the
    #    goal-buffer degrades, partially resetting rule utilities toward
    #    baseline. Models WM maintenance failure (Altmann & Gray, 2008).
    # ------------------------------------------------------------------
    if trial_state["consecutive_correct"] >= 5:
        if np.random.random() < set_loss_rate:
            decay = set_loss_strength
            for rule in RULES:
                key = f"select_{rule}"
                u = utilities.get(key, init_u)
                utilities[key] = u * (1.0 - decay) + init_u * decay
            trial_state["set_lost"] = True
            # Don't force random selection here -- let the degraded
            # utilities produce a natural (possibly wrong) choice
        else:
            trial_state["set_lost"] = False
    else:
        trial_state["set_lost"] = False

    # ------------------------------------------------------------------
    # 3. HYPOTHESIS TRACKING VIA DM RETRIEVAL
    #    After an error, the model uses declarative memory to recall
    #    which rules were already tried and failed.  Successfully recalled
    #    rules get their utility SUPPRESSED; untried rules get a BOOST.
    #    This implements the hypothesis-testing strategy observed in
    #    healthy adults (Barcelo & Knight, 2002).
    # ------------------------------------------------------------------
    boosted_utilities = dict(utilities)

    if trial_state.get("last_was_error", False):
        tried = trial_state.get("tried_since_switch", set())

        if tried:
            # Run pyactr DM retrieval to recall a failed rule
            recalled = _run_dm_retrieval(list(tried), params)

            if recalled:
                # Suppress the recalled (failed) rule
                key_bad = f"select_{recalled}"
                u_bad = boosted_utilities.get(key_bad, init_u)
                boosted_utilities[key_bad] = u_bad - hypothesis_boost

                # Boost untried alternatives
                untried = [r for r in RULES if r not in tried]
                for rule in untried:
                    key = f"select_{rule}"
                    u = boosted_utilities.get(key, init_u)
                    boosted_utilities[key] = u + hypothesis_boost

    # ------------------------------------------------------------------
    # 4. RUN PYACTR PRODUCTION SYSTEM
    #    Core decision: utility-based conflict resolution among the three
    #    rule-selection productions, followed by reward-driven update.
    # ------------------------------------------------------------------
    chosen, is_correct, new_utilities = _run_single_trial(
        boosted_utilities, correct_rule, params
    )

    # Merge: carry forward only the select_* utility changes that came
    # from pyactr's reward learning, removing the temporary boost/suppress.
    for rule in RULES:
        key = f"select_{rule}"
        delta = new_utilities[key] - boosted_utilities.get(key, init_u)
        base = utilities.get(key, init_u)
        utilities[key] = base + delta

    # Also carry forward feedback production utilities
    for fb in ["fb_correct", "fb_incorrect"]:
        utilities[fb] = new_utilities[fb]

    return chosen, is_correct, utilities, trial_state


# ============================================================================
# VARIANT MECHANISMS
# ============================================================================

def _apply_utility_decay(utilities, params, mechanisms):
    """
    Continuous utility decay toward baseline each trial.
    Models temporal context decay (Altmann & Gray, 2008): without
    ongoing reinforcement, rule preferences fade.
    """
    decay_rate = mechanisms.get("decay_rate", 0.015)
    init_u = params["initial_utility"]
    for rule in RULES:
        key = f"select_{rule}"
        u = utilities.get(key, init_u)
        utilities[key] = u + decay_rate * (init_u - u)
    return utilities


def _apply_asymmetric_lr_correction(utilities, old_utilities, is_correct, params, mechanisms):
    """
    Rescale pyactr's utility update based on outcome valence.
    Humans learn faster from punishment than reward (Bishara et al., 2010).
    We let pyactr compute delta with its alpha, then multiply by
    alpha_neg_mult for incorrect trials.
    """
    mult = mechanisms.get("alpha_neg_multiplier", 1.8) if not is_correct else 1.0
    if mult == 1.0:
        return utilities
    init_u = params["initial_utility"]
    for rule in RULES:
        key = f"select_{rule}"
        old_u = old_utilities.get(key, init_u)
        new_u = utilities.get(key, init_u)
        delta = new_u - old_u
        if abs(delta) > 1e-8:
            utilities[key] = old_u + delta * mult
    return utilities


def _apply_lose_shift(utilities, chosen_rule, is_correct, params, mechanisms):
    """
    After an error, directly boost the two non-chosen rules.
    Models the well-documented lose-shift heuristic in WCST
    (Heaton, 1993; Nyhus & Barcelo, 2009): humans actively redirect
    attention to alternatives after negative feedback.
    """
    if is_correct:
        return utilities
    boost = mechanisms.get("lose_shift_boost", 2.0)
    init_u = params["initial_utility"]
    for rule in RULES:
        if rule != chosen_rule:
            key = f"select_{rule}"
            utilities[key] = utilities.get(key, init_u) + boost
    return utilities


def _get_frustration_noise(base_noise, consecutive_errors, mechanisms):
    """
    After consecutive errors, temporarily increase exploration noise.
    Models the explore-exploit shift under uncertainty (Daw et al., 2006):
    when the current strategy is clearly failing, humans broaden their search.
    """
    threshold = mechanisms.get("frustration_threshold", 3)
    mult = mechanisms.get("frustration_noise_mult", 1.8)
    if consecutive_errors >= threshold:
        return base_noise * mult
    return base_noise


# ============================================================================
# SIMULATION RUNNERS
# ============================================================================

def run_wcst_simulation(params=None, seed=None, enhanced=False, mechanisms=None):
    """
    Run a complete 128-trial WCST simulation.

    Parameters
    ----------
    params : dict
        Model parameters.
    seed : int
        Random seed.
    enhanced : bool
        If True, use the enhanced model with three base mechanisms.
    mechanisms : dict or None
        Additional variant mechanisms to apply on top of enhanced model.
        Keys: "asymmetric_lr", "lose_shift", "frustration_explore", "utility_decay"
        plus their associated sub-parameters.
    """
    if params is None:
        params = get_default_params(enhanced=enhanced)
    if seed is not None:
        np.random.seed(seed)
    if mechanisms is None:
        mechanisms = {}

    utilities = {}
    consecutive_correct = 0
    consecutive_errors = 0
    current_category = 0
    current_rule_idx = 0
    current_rule = RULE_SEQUENCE[current_rule_idx]
    previous_rule = None
    correct_in_current_run = 0

    trial_data = []
    utility_traces = {"color": [], "shape": [], "number": []}
    rule_switch_trials = []

    # Enhanced model state
    trial_state = {
        "consecutive_correct": 0,
        "last_was_error": False,
        "tried_since_switch": set(),
        "set_lost": False,
        "lapse": False,
    }

    for trial_num in range(MAX_TRIALS):
        if current_category >= MAX_CATEGORIES:
            break

        # --- Pre-trial: utility decay ---
        if mechanisms.get("utility_decay"):
            utilities = _apply_utility_decay(utilities, params, mechanisms)

        for rule in RULES:
            utility_traces[rule].append(
                utilities.get(f"select_{rule}", params["initial_utility"])
            )

        # --- Snapshot utilities before trial (for asymmetric LR) ---
        pre_trial_utilities = dict(utilities)

        # --- Determine effective noise (frustration exploration) ---
        effective_params = dict(params)
        if mechanisms.get("frustration_explore"):
            effective_params["utility_noise"] = _get_frustration_noise(
                params["utility_noise"], consecutive_errors, mechanisms
            )

        # --- Run trial ---
        if enhanced:
            trial_state["consecutive_correct"] = consecutive_correct
            chosen_rule, is_correct, utilities, trial_state = (
                _run_enhanced_single_trial(
                    utilities, current_rule, effective_params, trial_state
                )
            )
            was_lapse = trial_state.get("lapse", False)
            was_set_lost = trial_state.get("set_lost", False)
        else:
            chosen_rule, is_correct, utilities = _run_single_trial(
                utilities, current_rule, effective_params
            )
            was_lapse = False
            was_set_lost = False

        # --- Post-trial: asymmetric learning rate correction ---
        if mechanisms.get("asymmetric_lr") and not was_lapse:
            utilities = _apply_asymmetric_lr_correction(
                utilities, pre_trial_utilities, is_correct, params, mechanisms
            )

        # --- Post-trial: lose-shift bias ---
        if mechanisms.get("lose_shift") and not was_lapse:
            utilities = _apply_lose_shift(
                utilities, chosen_rule, is_correct, params, mechanisms
            )

        # --- Classify errors ---
        is_perseverative_error = False
        is_perseverative_response = False
        if previous_rule is not None and chosen_rule == previous_rule:
            is_perseverative_response = True
            if not is_correct:
                is_perseverative_error = True

        failure_to_maintain = False
        if not is_correct and correct_in_current_run >= 5:
            failure_to_maintain = True

        trial_data.append({
            "trial": trial_num + 1,
            "category": current_category + 1,
            "correct_rule": current_rule,
            "chosen_rule": chosen_rule,
            "correct": is_correct,
            "perseverative_error": is_perseverative_error,
            "perseverative_response": is_perseverative_response,
            "failure_to_maintain": failure_to_maintain,
            "consecutive_correct": consecutive_correct,
            "lapse": was_lapse,
            "set_lost": was_set_lost,
            "trials_since_switch": (
                trial_num - rule_switch_trials[-1]
                if rule_switch_trials else trial_num
            ),
        })

        # --- Update state ---
        if is_correct:
            consecutive_correct += 1
            correct_in_current_run += 1
            consecutive_errors = 0
        else:
            consecutive_correct = 0
            correct_in_current_run = 0
            consecutive_errors += 1

        # Enhanced state bookkeeping
        if enhanced:
            trial_state["last_was_error"] = not is_correct
            if not is_correct:
                trial_state["tried_since_switch"].add(chosen_rule)

        # --- Category completion ---
        if consecutive_correct >= CRITERION:
            current_category += 1
            if current_category < MAX_CATEGORIES:
                previous_rule = current_rule
                current_rule_idx += 1
                current_rule = RULE_SEQUENCE[current_rule_idx]
                rule_switch_trials.append(trial_num + 1)
                if enhanced:
                    trial_state["tried_since_switch"] = set()
                    trial_state["last_was_error"] = False
            consecutive_correct = 0
            correct_in_current_run = 0
            consecutive_errors = 0

    # --- Summary statistics ---
    total_trials = len(trial_data)
    total_errors = sum(1 for t in trial_data if not t["correct"])
    perseverative_errors = sum(1 for t in trial_data if t["perseverative_error"])
    perseverative_responses = sum(1 for t in trial_data if t["perseverative_response"])
    non_perseverative_errors = total_errors - perseverative_errors
    fms = sum(1 for t in trial_data if t["failure_to_maintain"])

    trials_first = total_trials
    for i, t in enumerate(trial_data):
        if t["category"] == 2:
            trials_first = i + 1
            break
    if current_category == 0:
        trials_first = total_trials

    run_len = 0
    conceptual_count = 0
    for t in trial_data:
        if t["correct"]:
            run_len += 1
        else:
            if run_len >= 3:
                conceptual_count += run_len
            run_len = 0
    if run_len >= 3:
        conceptual_count += run_len
    clr_pct = (conceptual_count / total_trials * 100) if total_trials > 0 else 0

    post_switch_window = 15
    post_switch_accuracy = []
    if rule_switch_trials:
        for offset in range(post_switch_window):
            correct_at_offset = []
            for sw in rule_switch_trials:
                idx = sw + offset
                if idx < total_trials:
                    correct_at_offset.append(1.0 if trial_data[idx]["correct"] else 0.0)
            if correct_at_offset:
                post_switch_accuracy.append(np.mean(correct_at_offset))
            else:
                post_switch_accuracy.append(None)

    summary = {
        "categories_completed": current_category,
        "total_errors": total_errors,
        "perseverative_errors": perseverative_errors,
        "perseverative_responses": perseverative_responses,
        "non_perseverative_errors": non_perseverative_errors,
        "trials_first_category": trials_first,
        "failure_to_maintain_set": fms,
        "conceptual_level_responses": round(clr_pct, 1),
        "total_trials_used": total_trials,
        "percent_errors": round(total_errors / total_trials * 100, 1)
        if total_trials > 0 else 0,
        "percent_perseverative_errors": round(
            perseverative_errors / total_trials * 100, 1
        ) if total_trials > 0 else 0,
    }

    return {
        "summary": summary,
        "trial_data": trial_data,
        "utility_traces": {
            k: [round(v, 4) for v in vals]
            for k, vals in utility_traces.items()
        },
        "rule_switch_trials": rule_switch_trials,
        "post_switch_accuracy": [
            round(v, 4) if v is not None else None
            for v in post_switch_accuracy
        ],
        "params": params,
    }


def _aggregate_results(all_results):
    """Aggregate a list of single-simulation results into summary stats."""
    summary_keys = list(all_results[0]["summary"].keys())
    aggregated = {}
    for key in summary_keys:
        values = [r["summary"][key] for r in all_results]
        aggregated[key] = {
            "mean": round(float(np.mean(values)), 2),
            "sd": round(float(np.std(values, ddof=1)), 2) if len(values) > 1 else 0,
            "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2),
            "values": [round(float(v), 2) for v in values],
        }

    max_len = max(len(r["trial_data"]) for r in all_results)
    trial_accuracy = []
    for t_idx in range(max_len):
        correct_vals = []
        for r in all_results:
            if t_idx < len(r["trial_data"]):
                correct_vals.append(1.0 if r["trial_data"][t_idx]["correct"] else 0.0)
        trial_accuracy.append({
            "trial": t_idx + 1,
            "mean_accuracy": round(float(np.mean(correct_vals)), 4),
            "sd": round(float(np.std(correct_vals, ddof=1)), 4)
            if len(correct_vals) > 1 else 0,
            "n": len(correct_vals),
        })

    avg_utility_traces = {}
    for rule in RULES:
        traces = [r["utility_traces"][rule] for r in all_results]
        max_trace_len = max(len(t) for t in traces)
        avg_trace = []
        for t_idx in range(max_trace_len):
            vals = [t[t_idx] for t in traces if t_idx < len(t)]
            avg_trace.append(round(float(np.mean(vals)), 4))
        avg_utility_traces[rule] = avg_trace

    all_post_switch = [r["post_switch_accuracy"] for r in all_results]
    max_ps_len = max(len(ps) for ps in all_post_switch) if all_post_switch else 0
    avg_post_switch = []
    for idx in range(max_ps_len):
        vals = [ps[idx] for ps in all_post_switch if idx < len(ps) and ps[idx] is not None]
        avg_post_switch.append(round(float(np.mean(vals)), 4) if vals else None)

    return aggregated, trial_accuracy, avg_utility_traces, avg_post_switch, max_len, max_ps_len


def run_both_models(n_sims=30, params=None, base_seed=42):
    """
    Run BOTH basic and enhanced models and return aggregated results for
    side-by-side comparison.
    """
    if params is None:
        params = get_default_params(enhanced=True)

    start = time_module.time()

    # --- Basic model ---
    basic_results = []
    for i in range(n_sims):
        result = run_wcst_simulation(params=params, seed=base_seed + i, enhanced=False)
        basic_results.append(result)

    # --- Enhanced model ---
    enhanced_results = []
    for i in range(n_sims):
        result = run_wcst_simulation(params=params, seed=base_seed + i, enhanced=True)
        enhanced_results.append(result)

    elapsed = time_module.time() - start

    # Aggregate basic
    b_agg, b_trial_acc, b_util, b_ps, b_max_len, b_max_ps = _aggregate_results(basic_results)
    # Aggregate enhanced
    e_agg, e_trial_acc, e_util, e_ps, e_max_len, e_max_ps = _aggregate_results(enhanced_results)

    max_len = max(b_max_len, e_max_len)
    max_ps = max(b_max_ps, e_max_ps)
    human_post_switch = _generate_human_post_switch_curve(max_ps)
    human_trial_accuracy = _generate_human_trial_accuracy(max_len, basic_results)

    return {
        "basic": {
            "model_summary": b_agg,
            "trial_accuracy": b_trial_acc,
            "utility_traces": b_util,
            "post_switch_accuracy": b_ps,
            "individual_results": [{"summary": r["summary"]} for r in basic_results],
            "rmsd": _compute_rmsd(b_agg),
        },
        "enhanced": {
            "model_summary": e_agg,
            "trial_accuracy": e_trial_acc,
            "utility_traces": e_util,
            "post_switch_accuracy": e_ps,
            "individual_results": [{"summary": r["summary"]} for r in enhanced_results],
            "rmsd": _compute_rmsd(e_agg),
        },
        "human_data": HUMAN_DATA,
        "human_post_switch": human_post_switch,
        "human_trial_accuracy": human_trial_accuracy,
        "n_simulations": n_sims,
        "elapsed_seconds": round(elapsed, 2),
        "params": params,
    }


# ============================================================================
# VARIANT COMPARISON
# ============================================================================

VARIANT_DEFINITIONS = [
    {
        "id": "basic",
        "label": "Basic (utility learning only)",
        "short": "Basic",
        "enhanced": False,
        "mechanisms": {},
        "description": "Pure ACT-R utility learning. Three competing productions, reward-driven updates. No additional cognitive mechanisms.",
        "references": "Anderson & Lebiere (1998)",
        "color": "#4f8cff",
    },
    {
        "id": "enhanced",
        "label": "Enhanced (DM + lapse + decay)",
        "short": "Enhanced",
        "enhanced": True,
        "mechanisms": {},
        "description": "Adds hypothesis tracking via DM retrieval, attentional lapses, and set maintenance decay to the basic model.",
        "references": "Barcelo & Knight (2002); Altmann & Gray (2008)",
        "color": "#a78bfa",
    },
    {
        "id": "asymmetric_lr",
        "label": "+ Asymmetric Learning Rates",
        "short": "+AsymLR",
        "enhanced": True,
        "mechanisms": {
            "asymmetric_lr": True,
            "alpha_neg_multiplier": 1.8,
        },
        "description": "Punishment-driven utility updates are amplified 1.8x relative to rewards. Models the well-documented negativity bias in reinforcement learning.",
        "references": "Bishara et al. (2010); Frank et al. (2004)",
        "color": "#f87171",
    },
    {
        "id": "lose_shift",
        "label": "+ Lose-Shift Bias",
        "short": "+LoseShift",
        "enhanced": True,
        "mechanisms": {
            "lose_shift": True,
            "lose_shift_boost": 2.0,
        },
        "description": "After errors, the two non-chosen rules receive a direct utility boost. Models the active redirection of attention to alternatives after negative feedback.",
        "references": "Nyhus & Barcelo (2009); Heaton (1993)",
        "color": "#4ade80",
    },
    {
        "id": "frustration",
        "label": "+ Frustration Exploration",
        "short": "+Frustration",
        "enhanced": True,
        "mechanisms": {
            "frustration_explore": True,
            "frustration_threshold": 3,
            "frustration_noise_mult": 1.8,
        },
        "description": "After 3+ consecutive errors, utility noise increases 1.8x. Models the explore-exploit tradeoff shift under sustained uncertainty.",
        "references": "Daw et al. (2006); Cohen et al. (2007)",
        "color": "#fbbf24",
    },
    {
        "id": "utility_decay",
        "label": "+ Utility Decay",
        "short": "+Decay",
        "enhanced": True,
        "mechanisms": {
            "utility_decay": True,
            "decay_rate": 0.015,
        },
        "description": "All utilities decay 1.5% toward baseline each trial. Models temporal context degradation: without reinforcement, rule preferences fade.",
        "references": "Altmann & Gray (2008); Anderson (2007)",
        "color": "#fb923c",
    },
    {
        "id": "combined",
        "label": "Combined (all mechanisms)",
        "short": "Combined",
        "enhanced": True,
        "mechanisms": {
            "asymmetric_lr": True,
            "alpha_neg_multiplier": 1.3,
            "lose_shift": True,
            "lose_shift_boost": 0.8,
            "frustration_explore": True,
            "frustration_threshold": 4,
            "frustration_noise_mult": 1.3,
            "utility_decay": True,
            "decay_rate": 0.005,
        },
        "description": "All four mechanisms with carefully attenuated parameters. Tests whether multiple weak mechanisms produce better fit than one strong one.",
        "references": "Anderson (2007); Bishara et al. (2010)",
        "color": "#22d3ee",
    },
]


def _compute_rmsd(model_agg):
    """Compute RMSD between model and human data, normalized by human SD."""
    keys = ["categories_completed", "total_errors", "perseverative_errors",
            "non_perseverative_errors", "trials_first_category",
            "conceptual_level_responses", "failure_to_maintain_set"]
    ss = 0.0
    n = 0
    for k in keys:
        if k in model_agg and k in HUMAN_DATA:
            m = model_agg[k]["mean"]
            h = HUMAN_DATA[k]["mean"]
            sd = HUMAN_DATA[k]["sd"]
            if sd > 0:
                ss += ((m - h) / sd) ** 2
                n += 1
    return round(float(np.sqrt(ss / n)), 3) if n > 0 else None


def run_all_variants(n_sims=20, params=None, base_seed=42):
    """
    Run ALL model variants and return comparative results.

    Returns a dict with variant results sorted by RMSD (best first).
    """
    if params is None:
        params = get_default_params(enhanced=True)

    start = time_module.time()
    variant_results = []

    for vdef in VARIANT_DEFINITIONS:
        vid = vdef["id"]
        v_start = time_module.time()

        results = []
        for i in range(n_sims):
            r = run_wcst_simulation(
                params=params,
                seed=base_seed + i,
                enhanced=vdef["enhanced"],
                mechanisms=vdef["mechanisms"],
            )
            results.append(r)

        agg, trial_acc, util_traces, ps_acc, max_len, max_ps = _aggregate_results(results)
        rmsd = _compute_rmsd(agg)

        variant_results.append({
            "id": vid,
            "label": vdef["label"],
            "short": vdef["short"],
            "description": vdef["description"],
            "references": vdef["references"],
            "color": vdef["color"],
            "model_summary": agg,
            "rmsd": rmsd,
            "elapsed": round(time_module.time() - v_start, 2),
            "post_switch_accuracy": ps_acc,
            "trial_accuracy": trial_acc,
            "utility_traces": util_traces,
        })

    # Sort by RMSD (best first)
    variant_results.sort(key=lambda v: v["rmsd"] if v["rmsd"] is not None else 999)

    elapsed = time_module.time() - start

    # Generate human reference curves
    max_len = max(len(v["trial_accuracy"]) for v in variant_results)
    max_ps = max(
        len(v["post_switch_accuracy"]) for v in variant_results
    ) if variant_results else 0
    human_ps = _generate_human_post_switch_curve(max_ps)
    human_trial = _generate_human_trial_accuracy(max_len, [])

    return {
        "variants": variant_results,
        "human_data": HUMAN_DATA,
        "human_post_switch": human_ps,
        "human_trial_accuracy": human_trial,
        "n_simulations": n_sims,
        "elapsed_seconds": round(elapsed, 2),
        "params": params,
    }


# ============================================================================
# TEACHING COMPARISON (3 models only)
# ============================================================================

TEACHING_MODELS = [
    {
        "id": "basic",
        "label": "Basic Model",
        "short": "Basic",
        "enhanced": False,
        "mechanisms": {},
        "color": "#4f8cff",
    },
    {
        "id": "enhanced",
        "label": "Enhanced Model",
        "short": "Enhanced",
        "enhanced": True,
        "mechanisms": {},
        "color": "#a78bfa",
    },
    {
        "id": "full",
        "label": "Full Model",
        "short": "Full",
        "enhanced": True,
        "mechanisms": {
            "asymmetric_lr": True,
            "alpha_neg_multiplier": 1.8,
        },
        "color": "#f87171",
    },
]


def run_teaching_comparison(n_sims=30, params=None, base_seed=42):
    """
    Run the three teaching models: Basic, Enhanced, Full (Enhanced+AsymLR).
    Returns rich data for the teaching tab.
    """
    if params is None:
        params = get_default_params(enhanced=True)

    start = time_module.time()
    models = {}

    for mdef in TEACHING_MODELS:
        mid = mdef["id"]
        results = []
        for i in range(n_sims):
            r = run_wcst_simulation(
                params=params,
                seed=base_seed + i,
                enhanced=mdef["enhanced"],
                mechanisms=mdef["mechanisms"],
            )
            results.append(r)

        agg, trial_acc, util_traces, ps_acc, max_len, max_ps = _aggregate_results(results)

        models[mid] = {
            "label": mdef["label"],
            "short": mdef["short"],
            "color": mdef["color"],
            "model_summary": agg,
            "trial_accuracy": trial_acc,
            "utility_traces": util_traces,
            "post_switch_accuracy": ps_acc,
            "rmsd": _compute_rmsd(agg),
        }

    elapsed = time_module.time() - start

    # Reference curves
    all_lens = [len(models[m]["trial_accuracy"]) for m in models]
    all_ps = [len(models[m]["post_switch_accuracy"]) for m in models]
    max_len = max(all_lens) if all_lens else 128
    max_ps = max(all_ps) if all_ps else 15

    return {
        "models": models,
        "human_data": HUMAN_DATA,
        "human_post_switch": _generate_human_post_switch_curve(max_ps),
        "human_trial_accuracy": _generate_human_trial_accuracy(max_len, []),
        "n_simulations": n_sims,
        "elapsed_seconds": round(elapsed, 2),
        "params": params,
    }


# ============================================================================
# BAYESIAN OPTIMIZATION (Optuna + CRN + Composite Objective)
# ============================================================================

def _compute_postswitch_rmsd(model_ps, human_ps):
    """RMSD between model and human post-switch recovery curves."""
    n = min(len(model_ps), len(human_ps))
    if n == 0:
        return 0.0
    ss = 0.0
    count = 0
    for i in range(n):
        if model_ps[i] is not None and human_ps[i] is not None:
            ss += (model_ps[i] - human_ps[i]) ** 2
            count += 1
    return float(np.sqrt(ss / count)) if count > 0 else 0.0


def _compute_trial_curve_rmsd(model_trial_acc, human_trial_acc):
    """RMSD between model and human trial-by-trial accuracy curves."""
    n = min(len(model_trial_acc), len(human_trial_acc))
    if n == 0:
        return 0.0
    ss = 0.0
    count = 0
    for i in range(n):
        m_val = model_trial_acc[i]
        h_val = human_trial_acc[i]
        if m_val is not None and h_val is not None:
            if isinstance(m_val, dict):
                m_val = m_val.get("mean_accuracy", m_val)
            ss += (float(m_val) - float(h_val)) ** 2
            count += 1
    return float(np.sqrt(ss / count)) if count > 0 else 0.0


def _composite_objective(params, mechanisms, n_sims=3, fixed_seeds=None,
                         base_seed=42):
    """
    Composite objective combining three RMSD components:
      - summary_rmsd:    7 key summary statistics (normalized by human SD)
      - postswitch_rmsd: post-switch recovery curve shape
      - trial_rmsd:      trial-by-trial accuracy profile

    Uses Common Random Numbers (CRN): all evaluations use the same
    fixed seed sequence, dramatically reducing comparison noise.

    Returns (composite, summary_rmsd, postswitch_rmsd, trial_rmsd)
    """
    seeds = fixed_seeds if fixed_seeds is not None else [base_seed + i for i in range(n_sims)]

    results = []
    for s in seeds:
        r = run_wcst_simulation(
            params=params, seed=s, enhanced=True, mechanisms=mechanisms,
        )
        results.append(r)

    agg, trial_acc, _util, ps_acc, max_len, max_ps = _aggregate_results(results)

    # Component 1: Summary statistics RMSD
    summary_rmsd = _compute_rmsd(agg)
    if summary_rmsd is None:
        summary_rmsd = 10.0

    # Component 2: Post-switch curve RMSD
    human_ps = _generate_human_post_switch_curve(max_ps)
    postswitch_rmsd = _compute_postswitch_rmsd(ps_acc, human_ps)

    # Component 3: Trial-by-trial accuracy RMSD
    human_trial = _generate_human_trial_accuracy(max_len, [])
    model_trial_vals = [t["mean_accuracy"] if isinstance(t, dict) else t for t in trial_acc]
    trial_rmsd = _compute_trial_curve_rmsd(model_trial_vals, human_trial)

    # Weighted composite: summary stats most important, then curves
    composite = 0.50 * summary_rmsd + 0.30 * postswitch_rmsd + 0.20 * trial_rmsd

    return composite, summary_rmsd, postswitch_rmsd, trial_rmsd


def run_grid_search(n_trials=150, n_sims_trial=5, n_sims_final=50,
                    base_seed=42, progress_callback=None,
                    # Legacy params kept for API compat (ignored)
                    n_sims_coarse=None, n_sims_fine=None):
    """
    Bayesian optimization using Optuna's TPE sampler with:
      - Common Random Numbers (CRN) for noise-reduced comparisons
      - Composite objective (summary stats + post-switch curve + trial curve)
      - All 9 parameters searched simultaneously

    Returns the same structure as the old grid search for chart compatibility,
    plus new fields for the composite RMSD breakdown.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mechanisms = {"asymmetric_lr": True, "alpha_neg_multiplier": 1.8}
    defaults = get_default_params(enhanced=True)

    start = time_module.time()

    # CRN: Generate fixed seeds used by ALL evaluations
    rng = np.random.RandomState(base_seed)
    fixed_seeds = rng.randint(0, 100000, size=n_sims_trial).tolist()

    # Track all trials for the top-20 visualization
    all_trial_results = []

    def objective(trial):
        p = {
            "utility_noise":    trial.suggest_float("utility_noise", 0.3, 3.0),
            "utility_alpha":    trial.suggest_float("utility_alpha", 0.05, 0.8),
            "initial_utility":  trial.suggest_float("initial_utility", 2.0, 10.0),
            "correct_reward":   trial.suggest_float("correct_reward", 5.0, 20.0),
            "incorrect_reward": trial.suggest_float("incorrect_reward", -30.0, -3.0),
            "lapse_rate":       trial.suggest_float("lapse_rate", 0.0, 0.12),
            "set_loss_rate":    trial.suggest_float("set_loss_rate", 0.0, 0.08),
            "set_loss_strength":trial.suggest_float("set_loss_strength", 0.1, 1.0),
            "hypothesis_boost": trial.suggest_float("hypothesis_boost", 1.0, 10.0),
        }

        composite, s_rmsd, ps_rmsd, t_rmsd = _composite_objective(
            p, mechanisms, fixed_seeds=fixed_seeds,
        )

        # Store component RMSDs as user attributes
        trial.set_user_attr("summary_rmsd", round(s_rmsd, 4))
        trial.set_user_attr("postswitch_rmsd", round(ps_rmsd, 4))
        trial.set_user_attr("trial_rmsd", round(t_rmsd, 4))

        all_trial_results.append({
            "params": {k: round(v, 4) for k, v in p.items()},
            "rmsd": round(s_rmsd, 4),
            "composite": round(composite, 4),
            "postswitch_rmsd": round(ps_rmsd, 4),
            "trial_rmsd": round(t_rmsd, 4),
        })

        return composite

    # Create and run the study
    sampler = optuna.samplers.TPESampler(seed=base_seed, n_startup_trials=20)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="wcst_bayesian_opt",
    )

    # Seed with default params as a known baseline
    study.enqueue_trial({
        "utility_noise": defaults["utility_noise"],
        "utility_alpha": defaults["utility_alpha"],
        "initial_utility": defaults["initial_utility"],
        "correct_reward": defaults["correct_reward"],
        "incorrect_reward": defaults["incorrect_reward"],
        "lapse_rate": defaults["lapse_rate"],
        "set_loss_rate": defaults["set_loss_rate"],
        "set_loss_strength": defaults["set_loss_strength"],
        "hypothesis_boost": defaults["hypothesis_boost"],
    })

    study.optimize(objective, n_trials=n_trials)

    search_elapsed = round(time_module.time() - start, 1)

    # ---- Extract best params ----
    best_trial = study.best_trial
    best_params = {k: round(v, 4) for k, v in best_trial.params.items()}

    # ---- Final: Full evaluation with many sims ----
    # Use fresh seeds (not CRN) for unbiased final evaluation
    opt_results = []
    for i in range(n_sims_final):
        r = run_wcst_simulation(
            params=best_params, seed=base_seed + 10000 + i,
            enhanced=True, mechanisms=mechanisms,
        )
        opt_results.append(r)
    o_agg, o_trial, o_util, o_ps, o_max_len, o_max_ps = _aggregate_results(opt_results)

    # Run default for comparison (same fresh seeds)
    def_results = []
    for i in range(n_sims_final):
        r = run_wcst_simulation(
            params=defaults, seed=base_seed + 10000 + i,
            enhanced=True, mechanisms=mechanisms,
        )
        def_results.append(r)
    d_agg, d_trial, d_util, d_ps, d_max_len, d_max_ps = _aggregate_results(def_results)

    max_len = max(o_max_len, d_max_len)
    max_ps = max(o_max_ps, d_max_ps)

    elapsed = time_module.time() - start

    # Compute final composite breakdown for both models
    human_ps_curve = _generate_human_post_switch_curve(max_ps)
    human_trial_curve = _generate_human_trial_accuracy(max_len, [])

    opt_ps_rmsd = _compute_postswitch_rmsd(o_ps, human_ps_curve)
    opt_trial_rmsd = _compute_trial_curve_rmsd(
        [t["mean_accuracy"] for t in o_trial], human_trial_curve
    )
    def_ps_rmsd = _compute_postswitch_rmsd(d_ps, human_ps_curve)
    def_trial_rmsd = _compute_trial_curve_rmsd(
        [t["mean_accuracy"] for t in d_trial], human_trial_curve
    )

    # Build top-20 for visualization (sorted by composite)
    all_trial_results.sort(key=lambda x: x["composite"])
    top_results = all_trial_results[:20]

    # Build convergence curve (best composite so far at each trial)
    convergence = []
    best_so_far = float("inf")
    for i, t in enumerate(study.trials):
        if t.value is not None and t.value < best_so_far:
            best_so_far = t.value
        convergence.append(round(best_so_far, 4))

    return {
        "best_params": best_params,
        "default_params": defaults,
        "optimizer": "bayesian",
        "optimized": {
            "label": "Optimized",
            "short": "Optimized",
            "color": "#22d3ee",
            "model_summary": o_agg,
            "trial_accuracy": o_trial,
            "utility_traces": o_util,
            "post_switch_accuracy": o_ps,
            "rmsd": _compute_rmsd(o_agg),
            "postswitch_rmsd": round(opt_ps_rmsd, 4),
            "trial_rmsd": round(opt_trial_rmsd, 4),
        },
        "default_run": {
            "label": "Default Full",
            "short": "Default",
            "color": "#f87171",
            "model_summary": d_agg,
            "trial_accuracy": d_trial,
            "utility_traces": d_util,
            "post_switch_accuracy": d_ps,
            "rmsd": _compute_rmsd(d_agg),
            "postswitch_rmsd": round(def_ps_rmsd, 4),
            "trial_rmsd": round(def_trial_rmsd, 4),
        },
        "grid_top_20": top_results,
        "convergence": convergence,
        "total_combos_tested": len(study.trials),
        "human_data": HUMAN_DATA,
        "human_post_switch": _generate_human_post_switch_curve(max_ps),
        "human_trial_accuracy": _generate_human_trial_accuracy(max_len, []),
        "n_simulations_final": n_sims_final,
        "elapsed_seconds": round(elapsed, 2),
        "stage1_seconds": search_elapsed,
        "stage2_seconds": round(elapsed - search_elapsed, 1),
        "mechanisms": mechanisms,
        "search_config": {
            "n_trials": n_trials,
            "n_sims_per_trial": n_sims_trial,
            "n_sims_final": n_sims_final,
            "objective_weights": {"summary": 0.50, "postswitch": 0.30, "trial": 0.20},
            "method": "TPE (Tree-structured Parzen Estimator)",
            "crn_seeds": len(fixed_seeds),
        },
    }


# ============================================================================
# SYNTHETIC HUMAN DATA
# ============================================================================

def _generate_human_post_switch_curve(length):
    if length == 0:
        return []
    curve = []
    for i in range(length):
        if i == 0:   acc = 0.33
        elif i == 1: acc = 0.42
        elif i == 2: acc = 0.55
        elif i == 3: acc = 0.65
        elif i == 4: acc = 0.75
        elif i == 5: acc = 0.82
        elif i == 6: acc = 0.87
        elif i == 7: acc = 0.90
        else:        acc = min(0.95, 0.90 + 0.01 * (i - 7))
        curve.append(round(acc, 4))
    return curve


def _generate_human_trial_accuracy(max_len, model_results):
    typical_switches = [12, 24, 37, 50, 64, 78]
    accuracy = []
    for t in range(max_len):
        base = 0.92
        near_switch = False
        distance_from_switch = 999
        for sw in typical_switches:
            d = t - sw
            if 0 <= d < 8:
                near_switch = True
                distance_from_switch = min(distance_from_switch, d)

        if t < 5:
            acc = 0.33 + t * 0.12
        elif t < 12:
            acc = 0.75 + (t - 5) * 0.025
        elif near_switch:
            dip = {0: 0.35, 1: 0.45, 2: 0.58, 3: 0.70, 4: 0.78, 5: 0.84, 6: 0.88}
            acc = dip.get(distance_from_switch, 0.91)
        else:
            acc = base
        acc = max(0.1, min(1.0, acc + np.random.normal(0, 0.03)))
        accuracy.append(round(acc, 4))
    return accuracy


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

def get_default_params(enhanced=False):
    base = {
        "utility_noise": 1.5,
        "utility_alpha": 0.3,
        "initial_utility": 5.0,
        "correct_reward": 10.0,
        "incorrect_reward": -15.0,
    }
    if enhanced:
        base.update({
            "lapse_rate": 0.04,
            "set_loss_rate": 0.02,
            "set_loss_strength": 0.5,
            "hypothesis_boost": 4.0,
        })
    return base


def get_param_descriptions():
    return {
        "utility_noise": {
            "label": "Utility Noise (s)",
            "description": "Logistic noise in production selection. Higher = more exploration.",
            "min": 0.0, "max": 3.0, "step": 0.1, "default": 1.5,
            "group": "core",
        },
        "utility_alpha": {
            "label": "Learning Rate (alpha)",
            "description": "Speed of utility updates after feedback. Lower = frontal-patient profile.",
            "min": 0.01, "max": 1.0, "step": 0.01, "default": 0.3,
            "group": "core",
        },
        "initial_utility": {
            "label": "Initial Utility (U0)",
            "description": "Starting utility for all productions. Equal = no bias.",
            "min": 0.0, "max": 20.0, "step": 0.5, "default": 5.0,
            "group": "core",
        },
        "correct_reward": {
            "label": "Correct Reward (R+)",
            "description": "Reward for correct sorts. Drives rule acquisition.",
            "min": 0.0, "max": 30.0, "step": 0.5, "default": 10.0,
            "group": "core",
        },
        "incorrect_reward": {
            "label": "Incorrect Reward (R-)",
            "description": "Penalty for errors. More negative = faster rule abandonment.",
            "min": -30.0, "max": 0.0, "step": 0.5, "default": -15.0,
            "group": "core",
        },
        "lapse_rate": {
            "label": "Lapse Rate",
            "description": "Probability of attentional lapse per trial (random selection).",
            "min": 0.0, "max": 0.2, "step": 0.01, "default": 0.04,
            "group": "enhanced",
        },
        "set_loss_rate": {
            "label": "Set-Loss Rate",
            "description": "Probability of losing set after 5+ consecutive correct.",
            "min": 0.0, "max": 0.3, "step": 0.01, "default": 0.02,
            "group": "enhanced",
        },
        "set_loss_strength": {
            "label": "Set-Loss Strength",
            "description": "How much utility regresses toward baseline on set loss (0-1).",
            "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5,
            "group": "enhanced",
        },
        "hypothesis_boost": {
            "label": "Hypothesis Boost",
            "description": "Utility boost for untried rules after DM recall of a failure.",
            "min": 0.0, "max": 10.0, "step": 0.5, "default": 4.0,
            "group": "enhanced",
        },
    }


if __name__ == "__main__":
    print("Running all 7 model variants (15 sims each)...\n")
    data = run_all_variants(n_sims=15, base_seed=100)

    print(f"{'Rank':<5} {'RMSD':<8} {'Variant':<35} {'PE':>6} {'NPE':>6} {'FMS':>6} {'CLR%':>6}")
    print("-" * 80)
    for i, v in enumerate(data["variants"]):
        ms = v["model_summary"]
        print(f"{i+1:<5} {v['rmsd']:<8} {v['short']:<35} "
              f"{ms['perseverative_errors']['mean']:>6.1f} "
              f"{ms['non_perseverative_errors']['mean']:>6.1f} "
              f"{ms['failure_to_maintain_set']['mean']:>6.1f} "
              f"{ms['conceptual_level_responses']['mean']:>6.1f}")

    print(f"\n{'':5} {'':8} {'Human':35} "
          f"{HUMAN_DATA['perseverative_errors']['mean']:>6} "
          f"{HUMAN_DATA['non_perseverative_errors']['mean']:>6} "
          f"{HUMAN_DATA['failure_to_maintain_set']['mean']:>6} "
          f"{HUMAN_DATA['conceptual_level_responses']['mean']:>6}")
    print(f"\nElapsed: {data['elapsed_seconds']}s")
