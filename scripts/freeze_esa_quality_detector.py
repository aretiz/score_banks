
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("outputs/esa_real_event_readiness")
OUT = ROOT / "frozen_quality_detector"

REQUIRED = [
    Path("scripts/power_method_pilot/esa_probe_pilot.py"),
    Path("scripts/power_method_pilot/quality_signal_audit.py"),
    Path("outputs/esa_probe_features_v1/development_probe_features.csv"),
    Path("outputs/esa_probe_pilot_v1/protocol.json"),
    Path("outputs/esa_quality_signal_audit_v1.log"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def main():
    missing = [str(path) for path in REQUIRED if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required development files:\n"
            + "\n".join(missing)
        )

    log_text = REQUIRED[-1].read_text(
        encoding="utf-8",
        errors="replace",
    )

    marker = (
        "QUALITY SOURCE AUDIT COMPLETE. "
        "The sealed ESA evaluation set was not loaded."
    )

    if marker not in log_text:
        raise RuntimeError(
            "Quality-source audit completion marker is missing"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_version": "esa_quality_detector_v1",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_data": "development_only",
        "sealed_evaluation_used": False,
        "alpha": 0.05,
        "independent_unit": "ESA incident cluster or nominal window",
        "learner_contract": (
            "Use the unchanged logistic learner and preprocessing "
            "implemented by esa_probe_pilot.py"
        ),
        "fit_roles": {
            "nominal": [
                "audit_nominal_fit",
                "nll_validation",
                "paired_nominal_design",
            ],
            "alternative": [
                "anomaly_design",
            ],
        },
        "calibration_role": "nominal_calibration",
        "evaluation_roles": [
            "nominal_test",
            "anomaly_test",
        ],
        "threshold": {
            "method": "split_conformal",
            "complete_score_calibrated": True,
            "formula": "k=ceil((n_cal+1)*(1-alpha))",
        },
        "detectors": [
            {
                "name": "quality_W_primary",
                "status": "primary_confirmatory",
                "learner": "logistic",
                "feature_set": "W",
                "description": "observed-window quality only",
            },
            {
                "name": "quality_C_secondary",
                "status": "secondary_sensitivity",
                "learner": "logistic",
                "feature_set": "C",
                "description": "complete quality feature set",
            },
        ],
        "reporting": {
            "families": ["class_7", "class_3"],
            "class_7": "primary_confirmatory",
            "class_3": "secondary_exploratory",
            "metrics": [
                "nominal_test_fpr",
                "family_power",
                "paired_rejection_gain",
                "event_score_win_rate",
            ],
            "uncertainty_unit": "independent incident cluster",
            "no_test_retuning": True,
        },
        "development_evidence": {
            "W_leave_class_3_out_power": 0.8750,
            "W_leave_class_3_out_fpr": 0.0390,
            "W_leave_class_7_out_power": 0.6316,
            "W_leave_class_7_out_fpr": 0.0191,
            "C_leave_class_3_out_power": 0.8333,
            "C_leave_class_3_out_fpr": 0.0420,
            "C_leave_class_7_out_power": 0.6842,
            "C_leave_class_7_out_fpr": 0.0546,
        },
        "source_hashes": {
            str(path): sha256(path)
            for path in REQUIRED
        },
    }

    protocol_path = OUT / "protocol.json"

    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_path = OUT / "SHA256SUMS.txt"
    checksum_path.write_text(
        f"{sha256(protocol_path)}  {protocol_path}\n",
        encoding="utf-8",
    )

    print("=== FROZEN ESA QUALITY DETECTOR ===")
    print("primary: logistic W")
    print("secondary: logistic C")
    print("alpha: 0.05")
    print("sealed evaluation loaded: NO")
    print(f"protocol: {protocol_path}")
    print(f"checksum: {checksum_path}")
    print()
    print("FINAL QUALITY-DETECTOR SEAL: PASS")


if __name__ == "__main__":
    main()
