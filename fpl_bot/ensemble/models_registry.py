"""Single place that knows how to run all four models and tolerate any one
of them failing (e.g. Model 4's LLM backend not being configured) without
taking the other three -- or the ensemble -- down with it.
"""

import warnings

warnings.filterwarnings("ignore")

from fpl_bot.models.basic_stats import compute_scores as _model3_scores
from fpl_bot.models.evolutionary.live_apply import predict_current_squad_scores as _model1_scores
from fpl_bot.models.statistical_predictor import predict_next_gameweek as _model2_scores
from fpl_bot.qualitative.synthesizer import run_qualitative_review

MODEL_KEYS = ["model1", "model2", "model3", "model4"]
MODEL_LABELS = {
    "model1": "Evolutionary strategy",
    "model2": "Statistical predictor",
    "model3": "Basic stats",
    "model4": "Qualitative agent",
}


def compute_all_model_scores(qualitative_backend: str = "ollama") -> dict:
    """Returns {"scores": {key: [PlayerScore]}, "errors": {key: str},
    "model4_notes": [...], "model4_overall": str}."""
    scores = {}
    errors = {}

    try:
        scores["model3"] = _model3_scores()
    except Exception as e:  # noqa: BLE001 -- one model failing shouldn't sink the run
        errors["model3"] = str(e)

    try:
        scores["model2"] = _model2_scores()
    except Exception as e:
        errors["model2"] = str(e)

    try:
        scores["model1"] = _model1_scores()
    except Exception as e:
        errors["model1"] = str(e)

    model4_notes, model4_overall = [], ""
    try:
        review = run_qualitative_review(qualitative_backend, base_scores=scores.get("model3"))
        scores["model4"] = review["scores"]
        model4_notes = review["notes"]
        model4_overall = review["overall_notes"]
    except Exception as e:
        errors["model4"] = str(e)

    return {"scores": scores, "errors": errors, "model4_notes": model4_notes, "model4_overall": model4_overall}
