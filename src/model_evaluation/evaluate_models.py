import os
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI to SQLite database (same as training notebook)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

client = MlflowClient()

MODEL_NAME = "logistic_regression_model"


def _get_val_accuracy(model_version) -> float:
    """
    Prefer the metric stored as a model version tag (what we surface in the Model Registry UI).
    Fall back to the source run's metrics if the tag is missing.
    """
    tag_val = model_version.tags.get("best_val_accuracy")
    if tag_val is not None:
        return float(tag_val)

    # Fallback: pull metric from the run (handles older versions that didn't tag the model version)
    try:
        run = client.get_run(model_version.run_id)
        metrics = run.data.metrics or {}
        for key in ("best_val_accuracy", "best_accuracy", "val_accuracy"):
            if key in metrics:
                return float(metrics[key])
    except Exception:
        pass

    return float("-inf")


# 1. Get all versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    print(f"No model versions found for '{MODEL_NAME}'")
    exit(1)

# 2. Identify the "current model" (the newest registered version)
current_version = max(versions, key=lambda v: int(v.version))
current_acc = _get_val_accuracy(current_version)

if current_acc == float("-inf"):
    print(
        "Current model version does not have 'best_val_accuracy' as a tag and no suitable "
        "metric was found on the source run. Ensure training sets the model version tag "
        "or logs a metric named 'best_val_accuracy'."
    )
    print("Available tags on current version:", dict(current_version.tags))
    exit(1)

# 3. Find current Production model version(s)
production_versions = [v for v in versions if v.current_stage == "Production"]

print(f"Found {len(versions)} model version(s) for '{MODEL_NAME}'")
print(f"Current (latest) version: {current_version.version} (best_val_accuracy={current_acc})")

# Enforce invariant: at most one Production version (we'll clean up extras)
if len(production_versions) > 1:
    # Keep the one with the best accuracy as production; demote the rest to None
    production_versions_sorted = sorted(
        production_versions, key=_get_val_accuracy, reverse=True
    )
    keep_prod = production_versions_sorted[0]
    for extra in production_versions_sorted[1:]:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=extra.version, stage="None"
        )
        print(f"Demoted extra Production version {extra.version} -> None")
    production_versions = [keep_prod]

if not production_versions:
    # No production yet: promote the current version
    client.transition_model_version_stage(
        name=MODEL_NAME, version=current_version.version, stage="Production"
    )
    print(f"Promoted version {current_version.version} -> Production (no prior Production existed)")
    exit(0)

prod_version = production_versions[0]
prod_acc = _get_val_accuracy(prod_version)
print(f"Current Production version: {prod_version.version} (best_val_accuracy={prod_acc})")

# 4. Compare and swap if current is better
if int(prod_version.version) == int(current_version.version):
    print("Latest version is already in Production; no action needed.")
    exit(0)

if current_acc > prod_acc:
    # Promote new and demote old back to None to ensure only 1 Production version.
    client.transition_model_version_stage(
        name=MODEL_NAME, version=current_version.version, stage="Production"
    )
    client.transition_model_version_stage(
        name=MODEL_NAME, version=prod_version.version, stage="None"
    )
    print(
        f"Swapped Production: {prod_version.version} -> None, "
        f"{current_version.version} -> Production (improved {prod_acc} -> {current_acc})"
    )
else:
    print(
        f"Kept existing Production {prod_version.version}; "
        f"current version {current_version.version} is not better ({current_acc} <= {prod_acc})"
    )