"""Generate deterministic Phase 0 parity fixtures.

Run this after installing the local package:
  python tools/generate_phase0_fixtures.py
"""

import argparse
import json
from pathlib import Path

from fastwoe import FastWoe

DEFAULT_OUTPUT = Path("tests/fixtures/parity/phase0_v1.json")


def build_fixture() -> dict:
    binary_rows = [
        ["A", "x"],
        ["A", "y"],
        ["B", "x"],
        ["C", "z"],
        ["B", "y"],
        ["C", "x"],
    ]
    binary_targets = [1, 0, 0, 1, 0, 1]
    feature_names = ["cat", "bucket"]

    binary_model = FastWoe()
    binary_model.fit_matrix(binary_rows, binary_targets, feature_names=feature_names)

    multiclass_rows = binary_rows
    multiclass_labels = ["c0", "c1", "c2", "c0", "c1", "c2"]
    multiclass_model = FastWoe()
    multiclass_model.fit_matrix_multiclass(
        multiclass_rows, multiclass_labels, feature_names=feature_names
    )

    return {
        "fixture_version": "phase0_v1",
        "binary": {
            "rows": binary_rows,
            "targets": binary_targets,
            "feature_names": feature_names,
            "feature_names_out": binary_model.get_feature_names(),
            "transform_matrix": binary_model.transform_matrix(binary_rows),
            "predict_proba_matrix": binary_model.predict_proba_matrix(binary_rows),
            "predict_ci_matrix_alpha_0_05": binary_model.predict_ci_matrix(binary_rows, alpha=0.05),
        },
        "multiclass": {
            "rows": multiclass_rows,
            "labels": multiclass_labels,
            "feature_names": feature_names,
            "class_labels_out": multiclass_model.get_class_labels(),
            "feature_names_multiclass_out": multiclass_model.get_feature_names_multiclass(),
            "transform_matrix_multiclass": multiclass_model.transform_matrix_multiclass(
                multiclass_rows
            ),
            "predict_proba_matrix_multiclass": multiclass_model.predict_proba_matrix_multiclass(
                multiclass_rows
            ),
            "predict_ci_matrix_multiclass_alpha_0_05": (
                multiclass_model.predict_ci_matrix_multiclass(multiclass_rows, alpha=0.05)
            ),
            "predict_proba_matrix_class_c0": multiclass_model.predict_proba_matrix_class(
                multiclass_rows, "c0"
            ),
            "predict_ci_matrix_class_c0_alpha_0_05": multiclass_model.predict_ci_matrix_class(
                multiclass_rows,
                "c0",
                alpha=0.05,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 0 parity fixture JSON.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for fixture JSON.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture()
    output_path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote fixture: {output_path}")


if __name__ == "__main__":
    main()
