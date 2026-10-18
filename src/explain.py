"""
explain.py: what drives the departures forecast, using SHAP.
Run:  python -m src.explain
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.train import load_dataset, time_split


def get_explainer(bundle, background):
    """Background rows are the reference point SHAP measures against."""
    return shap.Explainer(bundle["model"].predict, background[bundle["features"]])


def main(n_explain=300):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(MODELS_DIR / "departures_model.joblib")
    train, test = time_split(load_dataset())

    background = train.sample(100, random_state=RANDOM_STATE)
    # Explain busy hours, where the forecast matters most.
    sample = test[test["departures"] > 0].sample(n_explain, random_state=RANDOM_STATE)

    explainer = get_explainer(bundle, background)
    shap_values = explainer(sample[bundle["features"]])

    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("What drives predicted departures")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close()

    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=150)
    plt.close()
    print(f"Saved SHAP plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()