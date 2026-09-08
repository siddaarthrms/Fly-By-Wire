from __future__ import annotations

import argparse
import os
from pathlib import Path

from caveclient import CAVEclient
from caveclient.base import AuthException


DATSTACK_NAME = "flywire_fafb_public"
# `flywire_fafb_public` materializations (e.g. v783) do not include PNI tables; use hierarchical annotations.
PNI_TABLE = "pni_drosophila_v6_v0"
DEFAULT_TABLE = "hierarchical_neuron_annotations"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_FILE = REPO_ROOT / "data" / "secrets" / "flywire_token.json"
POWER_CELL_TYPES = ["GF", "DNg01", "DNg02", "DNg03"]


def make_client() -> CAVEclient:
    # Prefer an explicit env override if provided.
    token_override = os.getenv("FLYWIRE_TOKEN_FILE", "").strip()
    token_path = Path(token_override) if token_override else DEFAULT_TOKEN_FILE
    token_path = token_path.resolve()

    try:
        if token_path.exists():
            return CAVEclient(DATSTACK_NAME, auth_token_file=str(token_path))
        # Fallback to default CAVE auth config if no token file exists.
        return CAVEclient(DATSTACK_NAME)
    except AuthException as exc:
        msg = (
            "FlyWire authentication failed.\n"
            f"Token file used: {token_path}\n"
            "Your token is likely missing, expired, or for a different account.\n\n"
            "Refresh token interactively:\n"
            "  from caveclient import CAVEclient\n"
            "  c = CAVEclient(server_address='https://global.flywire.ai')\n"
            "  c.auth.get_new_token(open=True)\n\n"
            "Then save it to:\n"
            f"  {DEFAULT_TOKEN_FILE}\n"
            "or set env var FLYWIRE_TOKEN_FILE to your token path."
        )
        raise RuntimeError(msg) from exc


def _latest_materialization_version(mt) -> int:
    """Highest non-expired materialization version for this datastack."""
    return int(mt.most_recent_version())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch GF / DNg power-brake neurons from FlyWire.")
    parser.add_argument(
        "--verify-tables",
        action="store_true",
        help="Print materialize table names for the latest version and exit.",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help=(
            f"Annotation table to query (default: {DEFAULT_TABLE}). "
            f"{PNI_TABLE!r} is not materialized on {DATSTACK_NAME!r}; use --verify-tables to list available names."
        ),
    )
    args = parser.parse_args(argv)

    client = make_client()
    mt = client.materialize
    latest_version = _latest_materialization_version(mt)
    client.version = latest_version

    if args.verify_tables:
        tables = mt.get_tables(version=latest_version)
        print(f"Tables for datastack {DATSTACK_NAME!r}, version {latest_version} ({len(tables)} total):")
        for name in sorted(tables):
            print(f"  {name}")
        return

    power_neurons = mt.query_table(
        args.table,
        filter_in_dict={"cell_type": POWER_CELL_TYPES},
        select_columns=["pt_root_id", "cell_type"],
        materialization_version=latest_version,
    )

    print(
        f"Found {len(power_neurons)} Power/Brake neurons "
        f"(table={args.table!r}, version={latest_version}).",
    )
    print(power_neurons.head().to_string(index=False))


if __name__ == "__main__":
    main()
