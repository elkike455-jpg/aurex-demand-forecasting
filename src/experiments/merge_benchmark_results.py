import os
import pandas as pd


def main():
    reports_dir = "reports"
    files = [
        "regime_benchmark_results.csv",
        "sarimax_benchmark_results.csv",
        "hurdle_benchmark_results.csv",
        "tar_benchmark_results.csv",
        "tsb_benchmark_results.csv",
    ]

    parts = []
    missing = []
    for name in files:
        path = os.path.join(reports_dir, name)
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
        else:
            missing.append(path)

    if missing:
        raise FileNotFoundError(f"Missing benchmark files: {missing}")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["dataset", "model"]).reset_index(drop=True)

    out_path = os.path.join(reports_dir, "model_comparison_3datasets.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
