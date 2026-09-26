"""Signal-level feature engineering from per-transaction history.
"""
import numpy as np
import pandas as pd

from .data import AMOUNT, DIRECTION, ID, SIGNAL_DATE, TX_TIME, TX_TYPE, TX_TYPES

WINDOWS_DAYS = [1, 3, 7, 14, 30, 60, 90]
LAST_K = [10, 50]
MONTH_BUCKETS = 6  # 6 x 30-day buckets over the ~180-day history
FEW_TX_THRESHOLD = 10
# miqdor_indeksi is clipped from below at ~-2.9123 (only incoming bank transfers reach it)
AMOUNT_FLOOR = -2.9
AMOUNT_BINS = [-np.inf, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
BURST_START_SECOND = 23 * 3600 + 57 * 60  # 23:57:00
# timing/recency feature groups that the trigger burst would distort; also computed on
# the burst-free behavioural history with an "h_" prefix
HIST_TIMING_PREFIXES = ("span_days", "days_since", "active_day", "max_tx_per_day", "std_tx_per_day",
                        "gap_", "share_gap", "tx_per_", "hour_", "w", "last", "bucket")


def _amount_stats(g, prefix: str, full: bool = False) -> pd.DataFrame:
    a = g[AMOUNT]
    out = pd.DataFrame({
        f"{prefix}_n": a.size(),
        f"{prefix}_mean": a.mean(),
        f"{prefix}_std": a.std(),
        f"{prefix}_max": a.max(),
    })
    if full:
        out[f"{prefix}_sum"] = a.sum()
        out[f"{prefix}_min"] = a.min()
        out[f"{prefix}_median"] = a.median()
        for q in (0.1, 0.25, 0.75, 0.9):
            out[f"{prefix}_q{int(q * 100)}"] = a.quantile(q)
        out[f"{prefix}_skew"] = a.skew()
    return out


def fit_type_stats(train_tx: pd.DataFrame) -> pd.DataFrame:
    """Global mean/std of miqdor_indeksi per (type, direction), fit on TRAIN transactions only."""
    return train_tx.groupby([TX_TYPE, DIRECTION])[AMOUNT].agg(type_mu="mean", type_sd="std")


def _prepare(tx: pd.DataFrame, type_stats: pd.DataFrame) -> pd.DataFrame:
    tx = tx.join(type_stats, on=[TX_TYPE, DIRECTION])
    # amount relative to what is normal for that transaction type and direction
    tx["amt_z"] = (tx[AMOUNT] - tx["type_mu"]) / tx["type_sd"]
    t = tx[TX_TIME]
    tx["days_before"] = (tx[SIGNAL_DATE] - t.dt.floor("D")).dt.days
    tx["hour"] = t.dt.hour
    tx["is_night"] = (tx["hour"] < 6).astype(np.int8)
    tx["is_weekend"] = (t.dt.dayofweek >= 5).astype(np.int8)
    tx["is_in"] = (tx[DIRECTION] == "kirim").astype(np.int8)
    for ty in TX_TYPES:
        tx[f"is_{ty}"] = (tx[TX_TYPE] == ty).astype(np.int8)
    tx["is_big2"] = (tx[AMOUNT] > 2).astype(np.int8)
    tx["is_big3"] = (tx[AMOUNT] > 3).astype(np.int8)
    tx["is_floor"] = (tx[AMOUNT] < AMOUNT_FLOOR).astype(np.int8)
    tx["amt_bin"] = pd.cut(tx[AMOUNT], AMOUNT_BINS, labels=False)
    # miqdor_indeksi behaves like a standardised log-amount; exp() gives a positive
    # amount proxy so inflow/outflow volumes can be summed and compared.
    tx["amt_exp"] = np.exp(tx[AMOUNT])
    same = tx[ID].eq(tx[ID].shift())
    gap = (t - t.shift()).dt.total_seconds() / 3600.0
    tx["gap_h"] = gap.where(same)
    prev_in = tx["is_in"].shift().where(same)
    tx["in_then_out_1h"] = ((prev_in == 1) & (tx["is_in"] == 0) & (tx["gap_h"] <= 1)).astype(np.int8)
    tx["in_then_out_24h"] = ((prev_in == 1) & (tx["is_in"] == 0) & (tx["gap_h"] <= 24)).astype(np.int8)
    similar = (tx[AMOUNT] - tx[AMOUNT].shift()).abs() < 0.2
    tx["passthrough_1h"] = (tx["in_then_out_1h"].astype(bool) & similar).astype(np.int8)
    tx["passthrough_24h"] = (tx["in_then_out_24h"].astype(bool) & similar).astype(np.int8)
    tx["rev_rank"] = tx.groupby(ID, sort=False).cumcount(ascending=False)
    return tx


def trigger_burst_mask(tx: pd.DataFrame) -> pd.Series:
    """Transactions stamped in the final 3 minutes of the day before (or of) the alert date,
    or exactly at 00:00:00 on the alert date.
 """
    t = tx[TX_TIME]
    days_before = (tx[SIGNAL_DATE] - t.dt.floor("D")).dt.days
    tod = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
    late = (days_before <= 1) & (tod >= BURST_START_SECOND)
    at_alert = t == tx[SIGNAL_DATE]
    return late | at_alert


def _burst_features(b: pd.DataFrame) -> pd.DataFrame:
    g = b.groupby(ID, sort=False)
    out = pd.DataFrame({
        "burst_n": g.size(),
        "burst_amt_mean": g[AMOUNT].mean(),
        "burst_amt_std": g[AMOUNT].std(),
        "burst_amt_max": g[AMOUNT].max(),
        "burst_amt_min": g[AMOUNT].min(),
        "burst_z_mean": g["amt_z"].mean(),
        "burst_share_in": g["is_in"].mean(),
        "burst_expsum_in": b[b["is_in"] == 1].groupby(ID, sort=False)["amt_exp"].sum(),
        "burst_expsum_out": b[b["is_in"] == 0].groupby(ID, sort=False)["amt_exp"].sum(),
        "burst_span_s": (g[TX_TIME].max() - g[TX_TIME].min()).dt.total_seconds(),
    })
    for ty in TX_TYPES:
        out[f"burst_share_{ty}"] = g[f"is_{ty}"].mean()
        out[f"burst_{ty}_mean"] = b[b[TX_TYPE] == ty].groupby(ID, sort=False)[AMOUNT].mean()
    return out


def build_features(tx: pd.DataFrame, signals: pd.DataFrame, type_stats: pd.DataFrame,
                   burst_block: bool = False, hist_timing: bool = False) -> pd.DataFrame:
    """Return one row per signal_id (in the order of `signals`) with engineered features.
    """
    parts = [_core_features(_prepare(tx, type_stats))]
    if burst_block or hist_timing:
        burst = trigger_burst_mask(tx)
    if hist_timing:
        hist_core = _core_features(_prepare(tx.loc[~burst].reset_index(drop=True), type_stats))
        timing = hist_core[[c for c in hist_core.columns if c.startswith(HIST_TIMING_PREFIXES)]]
        parts.append(timing.add_prefix("h_"))
    if burst_block:
        parts.append(_burst_features(_prepare(tx.loc[burst].reset_index(drop=True), type_stats)))
    feats = pd.concat(parts, axis=1)
    feats = signals[[ID, SIGNAL_DATE]].set_index(ID).join(feats, how="left")
    return _derive(feats, burst_block, hist_timing).reset_index()


def _core_features(tx: pd.DataFrame) -> pd.DataFrame:
    g = tx.groupby(ID, sort=False)
    parts = [_amount_stats(g, "amt", full=True)]

    z = g["amt_z"]
    zdf = pd.DataFrame({"z_mean": z.mean(), "z_std": z.std(), "z_median": z.median(),
                        "z_q10": z.quantile(0.1), "z_q90": z.quantile(0.9), "z_max": z.max()})
    for col, values in ((DIRECTION, ["kirim", "chiqim"]), (TX_TYPE, TX_TYPES)):
        for v in values:
            zdf[f"z_mean_{v}"] = tx[tx[col] == v].groupby(ID, sort=False)["amt_z"].mean()
    zdf["z_mean_last30d"] = tx[tx["days_before"] <= 30].groupby(ID, sort=False)["amt_z"].mean()
    zdf["z_mean_last50"] = tx[tx["rev_rank"] < 50].groupby(ID, sort=False)["amt_z"].mean()
    parts.append(zdf)

    n = g.size().rename("n_tx")
    base = pd.DataFrame({"n_tx": n})
    flag_cols = ["is_in", "is_night", "is_weekend", "is_big2", "is_big3", "is_floor",
                 "in_then_out_1h", "in_then_out_24h", "passthrough_1h", "passthrough_24h"] + \
                [f"is_{t}" for t in TX_TYPES]
    shares = g[flag_cols].mean().add_prefix("share_")
    counts = g[flag_cols].sum().add_prefix("cnt_")
    parts += [base, shares, counts]

    top5 = tx.groupby(ID, sort=False)[AMOUNT].nlargest(5).groupby(level=0).mean().rename("amt_top5_mean")
    parts.append(top5)

    # direction and type splits
    for col, values in ((DIRECTION, ["kirim", "chiqim"]), (TX_TYPE, TX_TYPES)):
        for v in values:
            sub = tx[tx[col] == v].groupby(ID, sort=False)
            s = _amount_stats(sub, f"{v}", full=False)
            s[f"{v}_sum"] = sub[AMOUNT].sum()
            s[f"{v}_expsum"] = sub["amt_exp"].sum()
            parts.append(s)
    for d in ["kirim", "chiqim"]:
        for ty in TX_TYPES:
            sub = tx[(tx[DIRECTION] == d) & (tx[TX_TYPE] == ty)].groupby(ID, sort=False)
            dt = pd.DataFrame({f"{d}_{ty}_n": sub.size(), f"{d}_{ty}_mean": sub[AMOUNT].mean()})
            if ty in ("karta", "bank_otkazmasi", "naqd"):
                dt[f"{d}_{ty}_std"] = sub[AMOUNT].std()
                for q in (0.1, 0.5, 0.9):
                    dt[f"{d}_{ty}_q{int(q * 100)}"] = sub[AMOUNT].quantile(q)
            parts.append(dt)

    # amount-distribution shape: share of tx per amount bin, overall and for the two big types
    for name, mask in (("all", slice(None)), ("bank", tx[TX_TYPE] == "bank_otkazmasi"),
                       ("karta", tx[TX_TYPE] == "karta")):
        sub_tx = tx.loc[mask]
        h = pd.crosstab(sub_tx[ID], sub_tx["amt_bin"], normalize="index")
        h = h.reindex(columns=range(len(AMOUNT_BINS) - 1), fill_value=0.0)
        h.columns = [f"hist_{name}_b{c}" for c in h.columns]
        parts.append(h)

    # time / activity profile
    first = g[TX_TIME].min()
    last = g[TX_TIME].max()
    sig = g[SIGNAL_DATE].first()
    days_hist = g["days_before"].max()
    day_counts = tx.groupby([ID, "days_before"], sort=False).size()
    dc = day_counts.groupby(level=0)
    time_df = pd.DataFrame({
        "span_days": (last - first).dt.total_seconds() / 86400.0,
        "days_since_first": (sig - first.dt.floor("D")).dt.days,
        "days_since_last": (sig - last.dt.floor("D")).dt.days,
        "active_days": dc.size(),
        "max_tx_per_day": dc.max(),
        "std_tx_per_day": dc.std(),
        "gap_mean_h": g["gap_h"].mean(),
        "gap_median_h": g["gap_h"].median(),
        "gap_min_h": g["gap_h"].min(),
        "gap_max_h": g["gap_h"].max(),
        "gap_std_h": g["gap_h"].std(),
        "hour_mean": g["hour"].mean(),
        "hour_std": g["hour"].std(),
    })
    time_df["active_day_share"] = time_df["active_days"] / (days_hist + 1)
    time_df["tx_per_day"] = n / (time_df["days_since_first"] + 1)
    time_df["tx_per_active_day"] = n / time_df["active_days"]
    tx["gap_lt_10m"] = (tx["gap_h"] < 1 / 6).astype(np.int8)
    time_df["share_gap_lt_10m"] = tx.groupby(ID, sort=False)["gap_lt_10m"].mean()
    parts.append(time_df)

    # recency windows vs full history
    overall_mean = g[AMOUNT].mean()
    for w in WINDOWS_DAYS:
        sub_tx = tx[tx["days_before"] <= w]
        sub = sub_tx.groupby(ID, sort=False)
        wdf = pd.DataFrame({
            f"w{w}_n": sub.size(),
            f"w{w}_mean": sub[AMOUNT].mean(),
            f"w{w}_max": sub[AMOUNT].max(),
            f"w{w}_share_in": sub["is_in"].mean(),
            f"w{w}_share_naqd": sub["is_naqd"].mean(),
            f"w{w}_share_xalqaro": sub["is_xalqaro"].mean(),
            f"w{w}_share_bank": sub["is_bank_otkazmasi"].mean(),
        })
        wdf[f"w{w}_mean_minus_all"] = wdf[f"w{w}_mean"] - overall_mean.reindex(wdf.index)
        parts.append(wdf)

    # last-K transactions vs full history
    for k in LAST_K:
        sub_tx = tx[tx["rev_rank"] < k]
        sub = sub_tx.groupby(ID, sort=False)
        kdf = pd.DataFrame({
            f"last{k}_mean": sub[AMOUNT].mean(),
            f"last{k}_std": sub[AMOUNT].std(),
            f"last{k}_max": sub[AMOUNT].max(),
            f"last{k}_share_in": sub["is_in"].mean(),
            f"last{k}_share_naqd": sub["is_naqd"].mean(),
            f"last{k}_share_xalqaro": sub["is_xalqaro"].mean(),
            f"last{k}_span_h": (sub[TX_TIME].max() - sub[TX_TIME].min()).dt.total_seconds() / 3600.0,
        })
        parts.append(kdf)

    # 30-day bucket counts (bucket 0 = most recent) to capture trend in activity
    tx["bucket"] = np.minimum(tx["days_before"] // 30, MONTH_BUCKETS - 1)
    bc = tx.groupby([ID, "bucket"], sort=False).size().unstack(fill_value=0)
    bc = bc.reindex(columns=range(MONTH_BUCKETS), fill_value=0)
    bc.columns = [f"bucket{b}_n" for b in bc.columns]
    parts.append(bc)
    bm = tx.groupby([ID, "bucket"], sort=False)[AMOUNT].mean().unstack()
    bm = bm.reindex(columns=range(MONTH_BUCKETS))
    bm.columns = [f"bucket{b}_mean" for b in bm.columns]
    parts.append(bm)
    return pd.concat(parts, axis=1)


def _derive(f: pd.DataFrame, burst_block: bool = False, hist_timing: bool = False) -> pd.DataFrame:
    """Ratios and fill defaults computed on the joined signal-level table."""
    eps = 1e-9
    count_like = [c for c in f.columns if c == "n_tx" or c.endswith("_n") or c.startswith("cnt_")
                  or c.startswith("share_") or c.endswith("_sum") or c.endswith("_expsum")
                  or c.startswith("burst_expsum")]
    f = f.copy()
    f[count_like] = f[count_like].fillna(0)
    new = {}
    new["no_tx"] = (f["n_tx"] == 0).astype(np.int8)
    new["few_tx"] = (f["n_tx"] < FEW_TX_THRESHOLD).astype(np.int8)
    if burst_block:
        new["burst_share_of_all"] = f["burst_n"] / (f["n_tx"] + eps)
        new["burst_amt_minus_hist"] = f["burst_amt_mean"] - f["amt_mean"]
        new["burst_z_minus_hist"] = f["burst_z_mean"] - f["z_mean"]
        new["burst_in_out_logratio"] = np.log((f["burst_expsum_in"] + 1) / (f["burst_expsum_out"] + 1))
        for ty in TX_TYPES:
            new[f"burst_{ty}_share_minus_hist"] = f[f"burst_share_{ty}"] - f[f"share_is_{ty}"]
    if hist_timing:
        new["h_recent_month_vs_prior"] = (f["h_bucket0_n"] + 1) / (
            f[[f"h_bucket{b}_n" for b in range(1, MONTH_BUCKETS)]].mean(axis=1) + 1)

    new["in_out_count_ratio"] = (f["kirim_n"] + 1) / (f["chiqim_n"] + 1)
    new["in_out_volume_logratio"] = np.log((f["kirim_expsum"] + 1) / (f["chiqim_expsum"] + 1))
    new["net_flow_share"] = (f["kirim_expsum"] - f["chiqim_expsum"]) / (f["kirim_expsum"] + f["chiqim_expsum"] + eps)
    new["card_vs_bank_ratio"] = (f["karta_n"] + 1) / (f["bank_otkazmasi_n"] + 1)
    new["cash_intl_share"] = f["share_is_naqd"] + f["share_is_xalqaro"]
    new["cash_share_of_volume"] = f["naqd_expsum"] / (f["kirim_expsum"] + f["chiqim_expsum"] + eps)
    new["intl_share_of_volume"] = f["xalqaro_expsum"] / (f["kirim_expsum"] + f["chiqim_expsum"] + eps)

    new["amt_spike_z"] = (f["amt_max"] - f["amt_mean"]) / (f["amt_std"] + eps)
    new["amt_range"] = f["amt_max"] - f["amt_min"]

    daily_rate = f["n_tx"] / (f["days_since_first"] + 1)
    for w in WINDOWS_DAYS:
        new[f"w{w}_share_of_tx"] = f[f"w{w}_n"] / (f["n_tx"] + eps)
        new[f"w{w}_rate_ratio"] = (f[f"w{w}_n"] / (w + 1)) / (daily_rate + eps)
        new[f"w{w}_naqd_minus_all"] = f[f"w{w}_share_naqd"] - f["share_is_naqd"]
        new[f"w{w}_xalqaro_minus_all"] = f[f"w{w}_share_xalqaro"] - f["share_is_xalqaro"]
        new[f"w{w}_in_minus_all"] = f[f"w{w}_share_in"] - f["share_is_in"]
    new["z_last30d_minus_all"] = f["z_mean_last30d"] - f["z_mean"]
    new["z_last50_minus_all"] = f["z_mean_last50"] - f["z_mean"]
    for k in LAST_K:
        new[f"last{k}_mean_minus_all"] = f[f"last{k}_mean"] - f["amt_mean"]
        new[f"last{k}_naqd_minus_all"] = f[f"last{k}_share_naqd"] - f["share_is_naqd"]
        new[f"last{k}_xalqaro_minus_all"] = f[f"last{k}_share_xalqaro"] - f["share_is_xalqaro"]

    older = f[[f"bucket{b}_n" for b in range(1, MONTH_BUCKETS)]].mean(axis=1)
    new["recent_month_vs_prior"] = (f["bucket0_n"] + 1) / (older + 1)
    b = np.arange(MONTH_BUCKETS)
    counts = f[[f"bucket{i}_n" for i in b]].to_numpy(dtype=float)
    bc = b - b.mean()
    new["activity_trend"] = (counts * -bc).sum(axis=1) / (bc ** 2).sum() / (counts.mean(axis=1) + 1)
    means = f[[f"bucket{i}_mean" for i in b]]
    new["amount_trend"] = means.bfill(axis=1).mul(-bc, axis=1).sum(axis=1) / (bc ** 2).sum()

    sig = f[SIGNAL_DATE]
    new["sig_month"] = sig.dt.month
    new["sig_dow"] = sig.dt.dayofweek
    new["sig_days_since_2025"] = (sig - pd.Timestamp("2025-01-01")).dt.days
    f = pd.concat([f, pd.DataFrame(new, index=f.index)], axis=1)
    return f.drop(columns=[SIGNAL_DATE])


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in (ID, "eskalatsiya")]
