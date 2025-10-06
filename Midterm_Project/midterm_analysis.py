#!/usr/bin/env python3
"""
Midterm COVID Analysis — Infection vs Vaccination Speeds
Author: <Your Name>
Course: CS320/CS110 (Fall 2025)
"""
from typing import Dict, Optional, Tuple, cast
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# ------------------------------- Configuration -------------------------------

DATA_DIRS = [
    "./data",
    "../data",
    os.path.dirname(os.path.abspath(__file__)),
    ".",
    "/mnt/data",
]
OUTPUT_DIR = "./output"

CASES_FILE = "covid_19_clean_complete.csv"
VAX_FILE = "country_vaccinations.csv"
WORLDOMETER_FILE = "worldometer_data.csv"

CASES_THRESHOLD = 10_000
VAX_THRESHOLD = 10_000
WINDOW_DAYS = 30
RANDOM_SEED = 42

# -------------------------- Utility: path resolution --------------------------

def find_first_existing_path(fname: str, directories) -> Optional[str]:
    for d in directories:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

# --------------------- Utility: country name standardization ------------------

COUNTRY_ALIASES: Dict[str, str] = {
    "US": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "Mainland China": "China",
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Czechia": "Czech Republic",
    "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Congo (Brazzaville)": "Republic of the Congo",
    "Taiwan*": "Taiwan",
    "Hong Kong SAR": "Hong Kong",
    "Macao SAR": "Macau",
    "Burma": "Myanmar",
    "Bahamas, The": "Bahamas",
    "Gambia, The": "Gambia",
    "Eswatini": "Swaziland",
    "Côte d'Ivoire": "Ivory Coast",
    "Viet Nam": "Vietnam",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Brunei Darussalam": "Brunei",
    "United States of America": "United States",
}

def std_country(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return COUNTRY_ALIASES.get(name, name)

# ------------------------------ Data Loaders ---------------------------------

def load_cases() -> pd.DataFrame:
    path = find_first_existing_path(CASES_FILE, DATA_DIRS)
    if not path:
        raise FileNotFoundError(f"{CASES_FILE} not found in {DATA_DIRS}")
    df = pd.read_csv(path, parse_dates=["Date"])

    # Standardize naming
    if "Country/Region" in df.columns:
        df.rename(columns={"Country/Region": "Country"}, inplace=True)
    if "Country_Region" in df.columns and "Country" not in df.columns:
        df.rename(columns={"Country_Region": "Country"}, inplace=True)
    if "Confirmed" not in df.columns:
        raise ValueError("Expected 'Confirmed' column in cases dataset.")
    df["Country"] = df["Country"].map(std_country)

    # Aggregate to country-date (summing provinces, if any)
    df_country = (
        df.groupby(["Country", "Date"], as_index=False)["Confirmed"]
          .sum()
          .sort_values(by=["Country", "Date"])
    )
    # Ensure cumulative non-decreasing
    df_country["Confirmed"] = df_country.groupby("Country")["Confirmed"].cummax()
    return df_country

def load_vaccinations() -> pd.DataFrame:
    path = find_first_existing_path(VAX_FILE, DATA_DIRS)
    if not path:
        raise FileNotFoundError(f"{VAX_FILE} not found in {DATA_DIRS}")
    df = pd.read_csv(path, parse_dates=["date"])

    # Standardize column names
    if "country" in df.columns:
        df.rename(columns={"country": "Country"}, inplace=True)
    if "date" in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)
    if "total_vaccinations" not in df.columns:
        raise ValueError("Expected 'total_vaccinations' column in vaccination dataset.")
    df["Country"] = df["Country"].map(std_country)

    vax = (
        df.groupby(["Country", "Date"], as_index=False)["total_vaccinations"]
          .max()
          .sort_values(by=["Country", "Date"])
    )
    vax["total_vaccinations"] = vax.groupby("Country")["total_vaccinations"].cummax()
    return vax

def load_worldometer() -> pd.DataFrame:
    path = find_first_existing_path(WORLDOMETER_FILE, DATA_DIRS)
    if not path:
        raise FileNotFoundError(f"{WORLDOMETER_FILE} not found in {DATA_DIRS}")
    df = pd.read_csv(path)

    # Standardize country column
    for cand in ["Country/Region", "Country", "country"]:
        if cand in df.columns:
            df.rename(columns={cand: "Country"}, inplace=True)
            break
    df["Country"] = df["Country"].map(std_country)

    # Try to find population & GDP per capita under multiple likely column names
    pop_cols = ["Population", "Population (2020)", "pop", "population"]
    gdp_pc_cols = ["GDP ($ per capita)", "GDP per capita", "GDP per capita (USD)", "gdp_per_capita", "GDP ($per capita)"]

    def first_existing(col_list):
        for c in col_list:
            if c in df.columns:
                return c
        return None

    pop_col = first_existing(pop_cols)
    gdp_pc_col = first_existing(gdp_pc_cols)

    if pop_col is None:
        raise ValueError("Could not find a population column in worldometer_data.csv")
    keep = ["Country", pop_col] + ([gdp_pc_col] if gdp_pc_col else [])
    df = df[keep].copy()
    df.rename(columns={pop_col: "Population"}, inplace=True)
    if gdp_pc_col:
        df.rename(columns={gdp_pc_col: "GDP_per_capita"}, inplace=True)
        if df["GDP_per_capita"].dtype == object:
            df["GDP_per_capita"] = (
                df["GDP_per_capita"].astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
    else:
        df["GDP_per_capita"] = np.nan

    if df["Population"].dtype == object:
        df["Population"] = (
            df["Population"].astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    return df

# --------------------------- Speed Metric Functions ---------------------------

def first_date_reaching_threshold(series: pd.Series, threshold: int) -> Optional[pd.Timestamp]:
    """Return the first date where cumulative series >= threshold; series index must be DateTimeIndex."""
    meets = series[series >= threshold]
    if meets.empty:
        return None
    return cast(pd.Timestamp, meets.index[0])

def compute_speed_per_million(
    cum: pd.Series, population: float, start_date: pd.Timestamp, window_days: int
) -> Optional[float]:
    """Compute avg daily increase per million over [start_date, start_date + window_days)."""
    if start_date is None or pd.isna(population) or population <= 0:
        return None
    end_date = start_date + pd.Timedelta(days=window_days)

    if end_date not in cum.index:
        after = cum.loc[start_date:]
        if after.empty:
            return None
        clip = cum.loc[start_date: start_date + pd.Timedelta(days=window_days)]
        if len(clip) < max(14, int(0.75 * window_days)):
            return None
        start_val = clip.iloc[0]
        end_val = clip.iloc[-1]
        days = (clip.index[-1] - clip.index[0]).days or 1
    else:
        start_val = cum.loc[start_date]
        end_val = cum.loc[end_date]
        days = window_days

    delta = max(0.0, float(end_val - start_val))
    per_day = delta / days
    per_million = per_day / (population / 1_000_000.0)
    return per_million

def build_country_speed_metrics(
    cases_df: pd.DataFrame,
    vax_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    cases_threshold: int = CASES_THRESHOLD,
    vax_threshold: int = VAX_THRESHOLD,
    window_days: int = WINDOW_DAYS
) -> pd.DataFrame:
    """Return a DataFrame with one row per country and computed speed metrics."""

    cases_pivot = (
        cases_df
        .set_index("Date")
        .groupby("Country")["Confirmed"]
        .apply(lambda s: s.asfreq("D").interpolate().cummax())
    )

    vax_pivot = (
        vax_df
        .set_index("Date")
        .groupby("Country")["total_vaccinations"]
        .apply(lambda s: s.asfreq("D").interpolate().cummax())
    )

    rows = []
    for country in sorted(set(cases_df["Country"]).union(set(vax_df["Country"]))):
        pop_row = demo_df.loc[demo_df["Country"] == country]
        if pop_row.empty:
            continue
        population = float(pop_row["Population"].iloc[0]) if not pd.isna(pop_row["Population"].iloc[0]) else np.nan
        gdp_pc = float(pop_row["GDP_per_capita"].iloc[0]) if not pd.isna(pop_row["GDP_per_capita"].iloc[0]) else np.nan

        cases_series = cases_pivot.loc[country] if country in cases_pivot.index else None
        vax_series = vax_pivot.loc[country] if country in vax_pivot.index else None

        inf_start = vacc_start = None
        inf_speed_pm = vacc_speed_pm = None

        if isinstance(cases_series, pd.Series):
            inf_start = first_date_reaching_threshold(cases_series, cases_threshold)
            if inf_start is not None:
                inf_speed_pm = compute_speed_per_million(cases_series, population, inf_start, window_days)

        if isinstance(vax_series, pd.Series):
            vacc_start = first_date_reaching_threshold(vax_series, vax_threshold)
            if vacc_start is not None:
                vacc_speed_pm = compute_speed_per_million(vax_series, population, vacc_start, window_days)

        rows.append({
            "Country": country,
            "Population": population,
            "GDP_per_capita": gdp_pc,
            "infection_threshold_date": inf_start,
            "vaccination_threshold_date": vacc_start,
            "infection_speed_pm": inf_speed_pm,
            "vaccination_speed_pm": vacc_speed_pm
        })

    return pd.DataFrame(rows)

# ------------------------------ Analysis Steps --------------------------------

def compute_correlations(df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    sub = df.dropna(subset=["infection_speed_pm", "vaccination_speed_pm"])
    if len(sub) < 5:
        return (np.nan, np.nan), (np.nan, np.nan)
    pearson_r, pearson_p = stats.pearsonr(sub["infection_speed_pm"], sub["vaccination_speed_pm"])
    spearman_r, spearman_p = stats.spearmanr(sub["infection_speed_pm"], sub["vaccination_speed_pm"])
    return (pearson_r, pearson_p), (spearman_r, spearman_p)

def run_regression(df: pd.DataFrame):
    """
    Vaccination speed ~ Infection speed + GDP per capita + log(population)
    Robust to missing GDP, drops remaining NaNs. Returns (model|None, r2|None, df_used).
    """
    sub = df.copy()
    sub = sub.dropna(subset=["vaccination_speed_pm", "infection_speed_pm", "Population"])
    sub = sub[sub["Population"] > 0].copy()

    sub.loc[:, "log_population"] = np.log(sub["Population"])

    if "GDP_per_capita" not in sub.columns:
        sub.loc[:, "GDP_per_capita"] = np.nan

    gdp_med = sub["GDP_per_capita"].median(skipna=True)
    if not np.isnan(gdp_med):
        sub.loc[:, "GDP_per_capita"] = sub["GDP_per_capita"].fillna(gdp_med)
        feature_cols = ["infection_speed_pm", "GDP_per_capita", "log_population"]
    else:
        feature_cols = ["infection_speed_pm", "log_population"]

    model_df = sub.dropna(subset=feature_cols + ["vaccination_speed_pm"]).copy()

    try:
        ensure_output_dir(OUTPUT_DIR)
        dropped = sub.index.difference(model_df.index)
        if len(dropped) > 0:
            sub.loc[dropped].to_csv(os.path.join(OUTPUT_DIR, "regression_rows_dropped.csv"), index=False)
    except Exception:
        pass

    if len(model_df) < 10:
        return None, None, model_df

    X = model_df[feature_cols].to_numpy(dtype=float)
    y = model_df["vaccination_speed_pm"].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return model, r2, model_df

# --------------------------------- Plotting -----------------------------------

def save_scatter(df: pd.DataFrame, outpath: str):
    sub = df.dropna(subset=["infection_speed_pm", "vaccination_speed_pm"])
    if len(sub) < 5:
        return
    plt.figure()
    plt.scatter(sub["infection_speed_pm"], sub["vaccination_speed_pm"])
    plt.xlabel("Infection speed (avg new cases / day per million)")
    plt.ylabel("Vaccination speed (avg new doses / day per million)")
    plt.title("Infection vs Vaccination Speeds (per million, first 30 days after threshold)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_hist(df: pd.DataFrame, col: str, outpath: str):
    sub = df[col].dropna()
    if len(sub) == 0:
        return
    plt.figure()
    plt.hist(sub, bins=30)
    plt.xlabel(col.replace("_", " "))
    plt.ylabel("Count of countries")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# ----------------------------------- Main -------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    ensure_output_dir(OUTPUT_DIR)

    cases = load_cases()
    vax = load_vaccinations()
    demo = load_worldometer()

    metrics = build_country_speed_metrics(
        cases, vax, demo,
        cases_threshold=CASES_THRESHOLD,
        vax_threshold=VAX_THRESHOLD,
        window_days=WINDOW_DAYS
    )

    metrics_path = os.path.join(OUTPUT_DIR, "speeds_by_country.csv")
    metrics.to_csv(metrics_path, index=False)

    missing_report = (
        metrics.assign(
            missing_infection_speed=metrics["infection_speed_pm"].isna(),
            missing_vaccination_speed=metrics["vaccination_speed_pm"].isna(),
            missing_gdp=metrics["GDP_per_capita"].isna(),
            missing_pop=metrics["Population"].isna(),
        )[["Country", "missing_infection_speed", "missing_vaccination_speed", "missing_gdp", "missing_pop"]]
    )
    missing_report.to_csv(os.path.join(OUTPUT_DIR, "missing_report.csv"), index=False)

    (pearson_r, pearson_p), (spearman_r, spearman_p) = compute_correlations(metrics)

    model, r2, reg_df = run_regression(metrics)

    save_scatter(metrics, os.path.join(OUTPUT_DIR, "infection_vs_vaccination_scatter.png"))
    save_hist(metrics, "infection_speed_pm", os.path.join(OUTPUT_DIR, "infection_speed_hist.png"))
    save_hist(metrics, "vaccination_speed_pm", os.path.join(OUTPUT_DIR, "vaccination_speed_hist.png"))

    print("="*72)
    print("Infection vs Vaccination Speed Analysis (per million per day)")
    print(f"Thresholds: cases >= {CASES_THRESHOLD}, vaccinations >= {VAX_THRESHOLD}; window = {WINDOW_DAYS} days")
    print("-"*72)
    print(f"Countries with both speeds computed: {metrics.dropna(subset=['infection_speed_pm','vaccination_speed_pm']).shape[0]}")
    print(f"Total countries (union): {metrics.shape[0]}")
    print("-"*72)
    if not np.isnan(pearson_r):
        print(f"Pearson r = {pearson_r:.3f}  (p = {pearson_p:.2e})")
        print(f"Spearman r = {spearman_r:.3f} (p = {spearman_p:.2e})")
    else:
        print("Not enough paired data to compute correlations.")
    if model is not None:
        coef_names = ["infection_speed_pm", "GDP_per_capita", "log_population"] \
            if "GDP_per_capita" in reg_df.columns else ["infection_speed_pm", "log_population"]
        coefs = model.coef_
        intercept = model.intercept_
        print("-"*72)
        print("Linear Regression: vaccination_speed_pm ~ " + " + ".join(coef_names))
        print(f"R^2 = {r2:.3f}")
        for name, val in zip(coef_names, coefs):
            print(f"  Coef[{name}] = {val:.6f}")
        print(f"  Intercept = {intercept:.6f}")
    else:
        print("-"*72)
        print("Not enough countries with complete data to fit regression (need >= 10).")

    print("-"*72)
    print("Saved outputs:")
    print(f"  - {metrics_path}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'missing_report.csv')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'infection_vs_vaccination_scatter.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'infection_speed_hist.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'vaccination_speed_hist.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'regression_rows_dropped.csv')} (if any)")
    print("="*72)

if __name__ == "__main__":
    main()
