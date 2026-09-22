from __future__ import annotations

import json

from src.labeling.labeler import export_ground_truth


if __name__ == "__main__":
    output, report = export_ground_truth()
    print("GROUND TRUTH DATASET")
    print("--------------------")
    print(f"Total labeled: {report['reviewed_articles']}")
    print(f"Positive: {report['positive_count']}")
    print(f"Neutral: {report['neutral_count']}")
    print(f"Negative: {report['negative_count']}")
    print(f"Percentages: {report['class_percentages']}")
    print(f"Validation: {report['validation_status']}")
    print(f"Output: {report['output']}")
    if report["reviewed_articles"] and min(report["class_percentages"].values()) < 10:
        print("WARNING: one class is underrepresented.")
