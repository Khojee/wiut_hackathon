"""Data loading and temporal hygiene for the alert-triage problem."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "fintech_data"

TARGET = "eskalatsiya"
ID = "signal_id"
SIGNAL_DATE = "signal_sanasi"
TX_TIME = "tranzaksiya_vaqti"
DIRECTION = "kirim_chiqim"
TX_TYPE = "tranzaksiya_turi"
AMOUNT = "miqdor_indeksi"

DIRECTIONS = ["kirim", "chiqim"]
TX_TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]


def load_signals(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{split}_signals.csv", parse_dates=[SIGNAL_DATE])
    df[ID] = df[ID].astype(str)
    return df


def load_transactions(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / f"{split}_transactions.parquet")
    df[ID] = df[ID].astype(str)
    return df


def load_sample_submission() -> pd.DataFrame:
    matches = sorted(DATA_DIR.glob("sample_submission*.csv"))
    if not matches:
        raise FileNotFoundError("sample_submission*.csv not found in " + str(DATA_DIR))
    return pd.read_csv(matches[0])


def attach_signal_date(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the signal date onto transactions and sort chronologically per signal.

    Validates the relationship is many-to-one so no transaction row is duplicated.
    """
    out = tx.merge(signals[[ID, SIGNAL_DATE]], on=ID, how="inner", validate="many_to_one")
    out = out.sort_values([ID, TX_TIME], kind="mergesort").reset_index(drop=True)
    return out


def temporal_report(tx: pd.DataFrame) -> dict:
    """Describe how transaction timestamps relate to the signal date (tx must have SIGNAL_DATE)."""
    day = tx[TX_TIME].dt.floor("D")
    next_day = tx[SIGNAL_DATE] + pd.Timedelta(days=1)
    return {
        "rows": len(tx),
        "after_signal_day": int((tx[TX_TIME] >= next_day).sum()),
        "on_signal_day": int((day == tx[SIGNAL_DATE]).sum()),
        "on_signal_day_exact_midnight": int((tx[TX_TIME] == tx[SIGNAL_DATE]).sum()),
        "max_days_before_signal": int((tx[SIGNAL_DATE] - day).dt.days.max()),
    }


def filter_pre_signal(tx: pd.DataFrame) -> pd.DataFrame:
    """Keep only transactions on or before the signal calendar date.

    signal_sanasi has day granularity, so a transaction during the alert day is
    treated as part of the history that could have triggered the alert.
    """
    keep = tx[TX_TIME] < tx[SIGNAL_DATE] + pd.Timedelta(days=1)
    return tx.loc[keep].reset_index(drop=True)


def load_split(split: str):
    """Return (signals, cleaned transactions with signal date attached, temporal report)."""
    signals = load_signals(split)
    tx = attach_signal_date(load_transactions(split), signals)
    report = temporal_report(tx)
    tx = filter_pre_signal(tx)
    report["rows_after_filter"] = len(tx)
    return signals, tx, report
