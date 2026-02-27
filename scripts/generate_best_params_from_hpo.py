"""
Generate `best_params.yaml` files from HPO `best_trial.json` results.

Behavior:
- Scans HPO results under `hpo/` or `models/hpo/` (auto-detected).
- For each (model, category, horizon) picks the asset's trial with lowest `best_value`.
- Writes `configs/models/<model>/<category>/<horizon>/best_params.yaml` (or `configs/<model>/...` if `configs/models/` is absent).
- Creates missing directories and overwrites existing YAML files.
- Skips missing or invalid `best_trial.json` files safely and prints a concise summary.

Uses only: os, json, yaml
"""

import os
import json
import yaml


def _find_hpo_root() -> str:
    """Return existing HPO root directory ('hpo' or 'models/hpo').
    Raises SystemExit if neither exists.
    """
    candidates = ["hpo", os.path.join("models", "hpo")]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise SystemExit("ERROR: no HPO directory found (checked: {}).".format(", ".join(candidates)))


def _find_configs_root() -> str:
    """Return configs root where model configs live.

    Prefer `configs/models/` (existing repo layout). If only `configs/` exists,
    use that. If neither exists, create `configs/models/` and return it.
    """
    preferred = os.path.join("configs", "models")
    if os.path.isdir(preferred):
        return preferred
    if os.path.isdir("configs"):
        return "configs"
    # create preferred layout for consistency
    os.makedirs(preferred, exist_ok=True)
    return preferred


def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # defensive: continue on bad files
        print(f"  - warning: failed to load JSON '{path}': {exc}")
        return None


def _write_yaml(path: str, data: dict) -> None:
    # Overwrite existing file; write params as flat mapping
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    hpo_root = _find_hpo_root()
    configs_root = _find_configs_root()

    # aggregated[(model, category, horizon)] = {
    #   'best_value': float,
    #   'best_params': dict,
    #   'asset': str,
    #   'trial_file': str
    # }
    aggregated = {}

    # Traverse expected structure: hpo/<model>/<category>/<asset>/<horizon>/best_trial.json
    for model in sorted(os.listdir(hpo_root)):
        model_dir = os.path.join(hpo_root, model)
        if not os.path.isdir(model_dir):
            continue
        for category in sorted(os.listdir(model_dir)):
            category_dir = os.path.join(model_dir, category)
            if not os.path.isdir(category_dir):
                continue
            for asset in sorted(os.listdir(category_dir)):
                asset_dir = os.path.join(category_dir, asset)
                if not os.path.isdir(asset_dir):
                    continue
                for horizon in sorted(os.listdir(asset_dir)):
                    horizon_dir = os.path.join(asset_dir, horizon)
                    if not os.path.isdir(horizon_dir):
                        continue

                    trial_path = os.path.join(horizon_dir, "best_trial.json")
                    if not os.path.isfile(trial_path):
                        # skip missing best_trial.json safely
                        continue

                    j = _load_json(trial_path)
                    if not j:
                        continue

                    if "best_params" not in j:
                        print(f"  - warning: 'best_params' missing in {trial_path}; skipping")
                        continue
                    if "best_value" not in j:
                        print(f"  - warning: 'best_value' missing in {trial_path}; skipping")
                        continue

                    try:
                        value = float(j["best_value"])
                    except Exception as exc:
                        print(f"  - warning: invalid best_value in {trial_path}: {exc}; skipping")
                        continue

                    key = (model, category, str(horizon))
                    current = aggregated.get(key)
                    # choose the lowest best_value across assets
                    if (current is None) or (value < current["best_value"]):
                        aggregated[key] = {
                            "best_value": value,
                            "best_params": j["best_params"],
                            "asset": asset,
                            "trial_file": trial_path,
                        }

    if not aggregated:
        print("No valid `best_trial.json` files were found under: '", hpo_root, "'.")
        return

    written = 0
    for (model, category, horizon), info in sorted(aggregated.items()):
        out_dir = os.path.join(configs_root, model, category, horizon)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "best_params.yaml")

        # Write the selected best_params (flat mapping) and overwrite if exists
        _write_yaml(out_file, info["best_params"])

        # Print required summary for each written file
        print("Model:", model)
        print("Category:", category)
        print("Horizon:", horizon)
        print("Selected trial:", info["asset"])
        print("Best value:", info["best_value"])
        print("Written:", out_file)
        print("-" * 40)
        written += 1

    print(f"Done — wrote {written} best_params.yaml file(s) to '{configs_root}'.")


if __name__ == "__main__":
    main()
