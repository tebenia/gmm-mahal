"""Create the configured Experiment 2 artifact and result directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment_2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_layout(config_path: str | Path) -> list[Path]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    baselines = config.get("baselines", {})
    created: set[Path] = set()

    for spec in baselines.values():
        for key in ("model_path", "shap_cache_dir", "value_selector_cache_dir"):
            path = project_path(spec[key])
            directory = path.parent if key == "model_path" else path
            directory.mkdir(parents=True, exist_ok=True)
            created.add(directory)

        result_root = project_path(spec["result_root"])
        for rate in spec["poison_rates"]:
            rate_tag = ("{:.4f}".format(float(rate))).rstrip("0").rstrip(".").replace(".", "p")
            for sampling in spec["sampling_strategies"]:
                directory = result_root / f"poison_rate_{rate_tag}" / sampling
                directory.mkdir(parents=True, exist_ok=True)
                created.add(directory)

    return sorted(created)


def main() -> None:
    directories = prepare_layout(parse_args().config)
    print(f"Prepared {len(directories)} Experiment 2 directories")
    for directory in directories:
        print(directory)


if __name__ == "__main__":
    main()
