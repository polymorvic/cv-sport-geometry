from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import tyro

from src.utils.const import TIMESTAMPT_STRING
from src.utils.func import get_point_weights


def run(root_test_dir: Path = Path("results/tests"), n_recent: int | None = None) -> None:
    """Generate and save a summary of recent test runs as a CSV file.

    Loads recent test result files from ``root_test_dir``, merges them into a
    DataFrame, computes aggregate statistics for each metric, appends summary
    rows, prints the result, and saves it as
    ``test_summary_<TIMESTAMPT_STRING>.csv`` in the same directory.
    TIMESTAMPT_STRING comes from const module

    Args:
        root_test_dir: Directory containing test run results.
        n_recent: Optional number of most recent runs to include.

    Returns:
        None. Prints and saves the summary CSV.
    """
    dfs = []
    tuple_colnames = []
    cols_to_remove = [
        "pic_index",
        "pic_name",
    ]
    i = 0
    subdirs = sorted(root_test_dir.iterdir(), reverse=True)
    for subdir in subdirs:
        if subdir.is_file():
            continue

        if n_recent and i >= n_recent:
            break

        temp_df = pd.read_csv(subdir / "test_df.csv")
        temp_df["summary"] = temp_df.apply(get_point_weights, axis=1)

        if i > 0:
            temp_df.drop(columns=cols_to_remove, inplace=True)

        for col in temp_df.columns:
            if col in cols_to_remove:
                tuple_colnames.append(("", col))
            else:
                tuple_colnames.append((subdir.stem, col))

        dfs.append(temp_df)
        i += 1

    merged_df = pd.concat(dfs, axis=1)
    merged_df.columns = pd.MultiIndex.from_tuples(tuple_colnames)

    merged_df.loc[len(merged_df)] = [""] * len(merged_df.columns)
    aggrs = {
        "max": lambda x: x.max(),
        "min": lambda x: x.min(),
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "median": lambda x: np.median(x),
    }
    rows = []
    for key, func in aggrs.items():
        row = []
        for col in merged_df.columns:
            if not col[0]:
                row.append("")
            elif col[1] == "summary":
                row.append(key)
            else:
                s_func = func(merged_df[col][:-1])
                row.append(s_func)
        rows.append(row)

    stats_rows = pd.DataFrame(rows, columns=merged_df.columns)
    merged_df = pd.concat([merged_df, stats_rows], ignore_index=True)
    merged_df.to_csv(root_test_dir / f"test_summary_{TIMESTAMPT_STRING}.csv", index=False)
    pprint(merged_df)


if __name__ == "__main__":
    tyro.cli(run)
