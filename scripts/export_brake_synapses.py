from __future__ import annotations

import pandas as pd

from fetch_power_neurons import (
    DEFAULT_TABLE,
    POWER_CELL_TYPES,
    REPO_ROOT,
    make_client,
)
from fetch_power_neurons import _latest_materialization_version


def main() -> None:
    client = make_client()
    mt = client.materialize
    version = _latest_materialization_version(mt)
    client.version = version

    power_neurons = mt.query_table(
        DEFAULT_TABLE,
        filter_in_dict={"cell_type": POWER_CELL_TYPES},
        select_columns=["pt_root_id", "cell_type"],
        materialization_version=version,
    )
    brake_ids = power_neurons["pt_root_id"].dropna().astype("int64").unique().tolist()

    brake_weights = mt.synapse_query(
        post_ids=brake_ids,
        materialization_version=version,
        synapse_table="synapses_nt_v1",
    )

    out_path = REPO_ROOT / "fly_brake_weights.csv"
    brake_weights.to_csv(out_path, index=False)
    print(f"Brake Weights Exported! Connections: {len(brake_weights)}")
    print(f"Wrote: {out_path}")
    print(f"Materialization version: {version}")
    print(f"Post neurons (DNg03 / power filter): {len(brake_ids)}")

    if "pre_pt_root_id" in brake_weights.columns:
        pre_ids = set(brake_weights["pre_pt_root_id"].dropna().astype("int64").unique())
        print(f"Unique upstream (pre) neurons: {len(pre_ids)}")

        eye_grid = REPO_ROOT / "data" / "mappings" / "fly_eye_grid_full.csv"
        if eye_grid.exists():
            motion_ids = set(pd.read_csv(eye_grid)["pt_root_id"].astype("int64"))
            overlap = pre_ids & motion_ids
            only_motion = len(overlap)
            only_other = len(pre_ids - motion_ids)
            print(
                "Overlap with fly_eye_grid_full.csv (retinotopy mapping): "
                f"{only_motion} upstream ids in-set, {only_other} not in that map."
            )


if __name__ == "__main__":
    main()
