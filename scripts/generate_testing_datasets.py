"""Regenerate synthetic Telco CSVs for dashboard upload demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "testing_datasets"
SIZES = (1_000, 1_200, 1_500, 1_800, 2_200, 3_000)
MISSING_FEATURES = (
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "tenure",
    "TechSupport",
)


def _choice(rng: np.random.Generator, options: tuple[str, ...], p: tuple[float, ...], n: int) -> np.ndarray:
    return rng.choice(options, size=n, p=p)


def synthesize(seed: int, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows
    gender = _choice(rng, ("Female", "Male"), (0.5, 0.5), n)
    senior = rng.binomial(1, 0.16, n)
    partner = _choice(rng, ("Yes", "No"), (0.48, 0.52), n)
    dependents = _choice(rng, ("Yes", "No"), (0.30, 0.70), n)
    tenure = np.rint(rng.beta(1.65, 1.42, n) * 72).astype(int).clip(0, 72)
    contract = _choice(rng, ("Month-to-month", "One year", "Two year"), (0.55, 0.21, 0.24), n)
    internet = _choice(rng, ("DSL", "Fiber optic", "No"), (0.34, 0.44, 0.22), n)
    phone = _choice(rng, ("Yes", "No"), (0.90, 0.10), n)

    def service(internet_required: bool = False) -> np.ndarray:
        result = _choice(rng, ("Yes", "No"), (0.38, 0.62), n)
        if internet_required:
            result = np.where(internet == "No", "No internet service", result)
        return result

    multiple_lines = _choice(rng, ("Yes", "No"), (0.49, 0.51), n)
    multiple_lines = np.where(phone == "No", "No phone service", multiple_lines)
    online_security = service(internet_required=True)
    online_backup = service(internet_required=True)
    device_protection = service(internet_required=True)
    tech_support = service(internet_required=True)
    streaming_tv = service(internet_required=True)
    streaming_movies = service(internet_required=True)
    paperless = _choice(rng, ("Yes", "No"), (0.59, 0.41), n)
    payment = _choice(
        rng,
        ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"),
        (0.34, 0.23, 0.22, 0.21),
        n,
    )

    base = np.select([internet == "Fiber optic", internet == "DSL"], [57.0, 39.0], default=20.0)
    base += np.where(phone == "Yes", 23.5, 0.0)
    base += (multiple_lines == "Yes") * 7.0
    for feature in (online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies):
        base += (feature == "Yes") * 4.2
    monthly = np.clip(base + rng.normal(0, 8.5, n), 18.0, 120.0).round(2)
    total = np.maximum(0, tenure * monthly + rng.normal(0, 18, n)).round(2)
    total[tenure == 0] = 0.0

    logit = (
        -2.2
        + (contract == "Month-to-month") * 0.72
        + (tenure < 12) * 0.72
        + (internet == "Fiber optic") * 0.20
        + (payment == "Electronic check") * 0.32
        + (tech_support == "No") * 0.30
        + (senior == 1) * 0.10
        - (partner == "Yes") * 0.16
        - (contract == "Two year") * 0.52
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < probability, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"SYN{seed:02d}{index:07d}" for index in range(1, n + 1)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )


def main() -> None:
    proper = OUTPUT / "proper_datasets"
    improper = OUTPUT / "improper_datasets"
    proper.mkdir(parents=True, exist_ok=True)
    improper.mkdir(parents=True, exist_ok=True)
    for folder in (proper, improper):
        for stale in folder.glob("*.csv"):
            stale.unlink()

    for index, rows in enumerate(SIZES, start=1):
        frame = synthesize(seed=731 + index * 97, rows=rows)
        filename = f"testing{index}.csv"
        frame.to_csv(proper / filename, index=False)
        frame.drop(columns=[MISSING_FEATURES[index - 1]]).to_csv(improper / filename, index=False)


if __name__ == "__main__":
    main()
