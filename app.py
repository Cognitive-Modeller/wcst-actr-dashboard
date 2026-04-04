"""
Flask webapp for the WCST ACT-R simulation dashboard.
Serves the interactive comparison of model vs. human data.

Supports:
  - Basic vs Enhanced comparison (fast)
  - Full variant comparison: 7 model variants ranked by RMSD
  - Teaching mode: 3-model progression with narrative explanations
"""

from flask import Flask, render_template, jsonify, request
from wcst_model import (
    run_both_models,
    run_all_variants,
    run_teaching_comparison,
    run_grid_search,
    get_default_params,
    get_param_descriptions,
    HUMAN_DATA,
)

app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template(
        "index.html",
        params=get_default_params(enhanced=True),
        param_descriptions=get_param_descriptions(),
    )


@app.route("/api/run", methods=["POST"])
def run_simulation():
    """Run both WCST model variants and return results as JSON."""
    data = request.get_json(force=True) if request.is_json else {}

    params = get_default_params(enhanced=True)
    for key in params:
        if key in data:
            params[key] = float(data[key])

    n_sims = int(data.get("n_simulations", 30))
    n_sims = max(5, min(100, n_sims))

    base_seed = int(data.get("seed", 42))

    results = run_both_models(
        n_sims=n_sims, params=params, base_seed=base_seed
    )

    return jsonify(results)


@app.route("/api/variants", methods=["POST"])
def run_variants():
    """Run ALL 7 model variants and return ranked comparison."""
    data = request.get_json(force=True) if request.is_json else {}

    params = get_default_params(enhanced=True)
    for key in params:
        if key in data:
            params[key] = float(data[key])

    n_sims = int(data.get("n_simulations", 15))
    n_sims = max(5, min(50, n_sims))

    base_seed = int(data.get("seed", 42))

    results = run_all_variants(
        n_sims=n_sims, params=params, base_seed=base_seed
    )

    return jsonify(results)


@app.route("/api/teaching", methods=["POST"])
def run_teaching():
    """Run 3 teaching models (Basic, Enhanced, Full) and return results."""
    data = request.get_json(force=True) if request.is_json else {}

    params = get_default_params(enhanced=True)
    for key in params:
        if key in data:
            params[key] = float(data[key])

    n_sims = int(data.get("n_simulations", 30))
    n_sims = max(5, min(100, n_sims))

    base_seed = int(data.get("seed", 42))

    results = run_teaching_comparison(
        n_sims=n_sims, params=params, base_seed=base_seed
    )

    return jsonify(results)


@app.route("/api/optimize", methods=["POST"])
def run_optimize():
    """Run Bayesian optimization (Optuna + CRN + composite objective)."""
    data = request.get_json(force=True) if request.is_json else {}

    n_trials = int(data.get("n_trials", 150))
    n_trials = max(20, min(500, n_trials))

    n_sims_trial = int(data.get("n_sims_trial", 5))
    n_sims_trial = max(3, min(15, n_sims_trial))

    n_final = int(data.get("n_sims_final", 50))
    n_final = max(10, min(100, n_final))

    base_seed = int(data.get("seed", 42))

    results = run_grid_search(
        n_trials=n_trials,
        n_sims_trial=n_sims_trial,
        n_sims_final=n_final,
        base_seed=base_seed,
    )

    return jsonify(results)


@app.route("/api/defaults")
def defaults():
    """Return default parameters and human data."""
    return jsonify({
        "params": get_default_params(enhanced=True),
        "param_descriptions": get_param_descriptions(),
        "human_data": HUMAN_DATA,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
