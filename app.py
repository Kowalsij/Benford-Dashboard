import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Benford's Law Dashboard", layout="wide")
st.title("Benford's Law Analysis Dashboard")
st.write(
    "Upload a CSV or Excel file, filter the data, and test whether a numeric column "
    "follows Benford's Law using the Kolmogorov-Smirnov (KS) statistic."
)

# ---------------------------------------------------------------
# 1. File upload (CSV or Excel)
# ---------------------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or Excel file(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True
)


@st.cache_data
def load_file(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


if not uploaded_files:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

frames = []
for f in uploaded_files:
    temp = load_file(f.getvalue(), f.name).copy()
    # Standardize column names so files match regardless of case/spacing
    # e.g. "payment_amount", "Payment Amount", "PAYMENT_AMOUNT" -> "PAYMENT_AMOUNT"
    temp.columns = (
        temp.columns.astype(str).str.strip().str.upper()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    temp = temp.loc[:, ~temp.columns.duplicated()]
    if len(uploaded_files) > 1:
        temp["source_file"] = f.name
    frames.append(temp)
df = pd.concat(frames, ignore_index=True)

# Try to convert text columns that look like dates
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = converted

numeric_cols = df.select_dtypes(include="number").columns.tolist()
date_cols = df.select_dtypes(include="datetime").columns.tolist()
cat_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]

if not numeric_cols:
    st.error("The uploaded file has no numeric columns to analyze.")
    st.stop()

st.subheader("Raw Data Preview")
st.write(f"{len(df):,} rows x {df.shape[1]} columns")
st.dataframe(df.head(100), width="stretch")

# ---------------------------------------------------------------
# 2. Filters (three criteria: category, numeric range, date range)
# ---------------------------------------------------------------
st.sidebar.header("Filters")
filtered = df.copy()

# Filter 1: categorical
if cat_cols:
    cat_col = st.sidebar.selectbox("Filter 1 - Category column", ["(none)"] + cat_cols)
    if cat_col != "(none)":
        options = sorted(filtered[cat_col].dropna().astype(str).unique())
        chosen = st.sidebar.multiselect(f"Values of {cat_col}", options, default=options)
        filtered = filtered[filtered[cat_col].astype(str).isin(chosen)]

# Filter 2: numeric range
num_filter_col = st.sidebar.selectbox("Filter 2 - Numeric column", ["(none)"] + numeric_cols)
if num_filter_col != "(none)":
    lo, hi = float(df[num_filter_col].min()), float(df[num_filter_col].max())
    if lo < hi:
        rng = st.sidebar.slider(f"Range of {num_filter_col}", lo, hi, (lo, hi))
        filtered = filtered[filtered[num_filter_col].between(rng[0], rng[1])]

# Filter 3: date range
if date_cols:
    date_col = st.sidebar.selectbox("Filter 3 - Date column", ["(none)"] + date_cols)
    if date_col != "(none)":
        dmin, dmax = df[date_col].min().date(), df[date_col].max().date()
        dates = st.sidebar.date_input(f"Range of {date_col}", (dmin, dmax), dmin, dmax)
        if isinstance(dates, (list, tuple)) and len(dates) == 2:
            filtered = filtered[
                filtered[date_col].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1))
            ]

# Extra filter: minimum absolute value (common in Benford work to drop tiny amounts)
min_abs = st.sidebar.number_input("Exclude values with absolute value below", value=0.0, min_value=0.0)

st.sidebar.write(f"Rows after filtering: **{len(filtered):,}**")

# ---------------------------------------------------------------
# 3. Column of interest + period comparison setup
# ---------------------------------------------------------------
st.subheader("Column of Interest")
target = st.selectbox("Choose the numeric column to test", numeric_cols)


def clean_values(series):
    """Positive absolute values used for the Benford test."""
    v = pd.to_numeric(series, errors="coerce").dropna().abs()
    return v[(v > 0) & (v >= min_abs)]


st.subheader("Period Comparison")
compare = False
if date_cols:
    compare = st.toggle("Compare two time periods (e.g. 05 2025 vs 05 2026)", value=True)
else:
    st.info("No date column was found, so the analysis uses all filtered data.")

raw_groups = {}  # group label -> raw target values (for summary stats)
if compare:
    c1, c2 = st.columns(2)
    period_col = c1.selectbox("Date column to compare by", date_cols)
    granularity = c2.radio("Compare by", ["Month & Year (MM YYYY)", "Year (YYYY)"], horizontal=True)
    fmt, freq = ("%m %Y", "M") if granularity.startswith("Month") else ("%Y", "Y")

    period_labels = filtered[period_col].dt.to_period(freq).dt.strftime(fmt)
    periods = sorted(filtered[period_col].dropna().dt.to_period(freq).unique())
    labels = [p.strftime(fmt) for p in periods]

    if len(labels) < 2:
        st.warning("The filtered data covers fewer than two periods, so it is analyzed as one group.")
        compare = False
    else:
        # Default: first period that has the same month one year later (e.g. 05 2025 -> 05 2026)
        idx_a, idx_b = 0, len(labels) - 1
        for i, p in enumerate(periods):
            match = p + (12 if freq == "M" else 1)
            if match in periods:
                idx_a, idx_b = i, periods.index(match)
                break
        period_a = c1.selectbox("Period A", labels, index=idx_a)
        period_b = c2.selectbox("Period B", labels, index=idx_b)
        if period_a == period_b:
            st.warning("Choose two different periods to compare.")
            st.stop()
        for lab in (period_a, period_b):
            raw_groups[lab] = filtered.loc[period_labels == lab, target]

if not compare:
    raw_groups["All filtered data"] = filtered[target]

groups = {g: clean_values(s) for g, s in raw_groups.items()}
for g, v in groups.items():
    if len(v) == 0:
        st.error(f"No positive values of {target} remain for {g}.")
        st.stop()

# Summary statistics (one column per period)
summary = pd.DataFrame({g: s.describe() for g, s in raw_groups.items()})
summary.loc["values used in Benford test"] = [len(v) for v in groups.values()]
st.subheader(f"Summary Statistics - {target}")
st.dataframe(summary.style.format("{:,.2f}"), width="stretch")


# ---------------------------------------------------------------
# 4. Benford analysis
# ---------------------------------------------------------------
def first_digits(x, n_digits):
    """Return the first n digits of each positive number."""
    x = np.asarray(x, dtype=float)
    if n_digits == 2:
        x = x[x >= 10]  # need at least two digits
    exponent = np.floor(np.log10(x)) - (n_digits - 1)
    return np.floor(x / 10.0 ** exponent + 1e-9).astype(int)


def benford_table(x, n_digits):
    digits = np.arange(1, 10) if n_digits == 1 else np.arange(10, 100)
    d = first_digits(x, n_digits)
    counts = pd.Series(d).value_counts().reindex(digits, fill_value=0)
    n = counts.sum()
    table = pd.DataFrame({
        "Digit": digits,
        "Count": counts.values,
        "Actual": counts.values / n if n else 0,
        "Benford": np.log10(1 + 1 / digits),
    })
    table["Difference"] = table["Actual"] - table["Benford"]
    return table, n


CRITICAL = {"0.10": 1.22, "0.05": 1.36, "0.01": 1.63}


def ks_test(table, n, alpha_label="0.05"):
    """KS statistic with NumPy: largest gap between the two cumulative distributions."""
    actual_cdf = np.cumsum(table["Actual"].to_numpy())
    benford_cdf = np.cumsum(table["Benford"].to_numpy())
    ks_stat = np.max(np.abs(actual_cdf - benford_cdf))
    critical = CRITICAL[alpha_label] / np.sqrt(n)
    return ks_stat, critical, actual_cdf, benford_cdf


def ks_two_groups(cdf_a, cdf_b, n_a, n_b, alpha_label="0.05"):
    """Two-sample KS: compares Period A's distribution directly to Period B's."""
    ks_stat = np.max(np.abs(cdf_a - cdf_b))
    critical = CRITICAL[alpha_label] * np.sqrt((n_a + n_b) / (n_a * n_b))
    return ks_stat, critical


alpha = st.radio("Significance level (alpha)", ["0.10", "0.05", "0.01"], index=1, horizontal=True)

tests = [(1, "First Digit"), (2, "First Two Digits")]
results = {}      # test label -> group label -> result dict
between = {}      # test label -> (ks, critical) for Period A vs Period B
for k, label in tests:
    results[label] = {}
    for g, v in groups.items():
        table, n = benford_table(v, k)
        if n == 0:
            continue
        ks_stat, critical, a_cdf, b_cdf = ks_test(table, n, alpha)
        results[label][g] = dict(table=table, n=n, ks=ks_stat, crit=critical, a_cdf=a_cdf, b_cdf=b_cdf)
    if len(results[label]) == 2:
        (ga, ra), (gb, rb) = results[label].items()
        between[label] = ks_two_groups(ra["a_cdf"], rb["a_cdf"], ra["n"], rb["n"], alpha)
results = {k: v for k, v in results.items() if v}

# ---------------------------------------------------------------
# 5. Visualizations + statistical results
# ---------------------------------------------------------------
COLORS = ["#4C78A8", "#F58518"]
figures = {}
tabs = st.tabs(list(results.keys()))
for tab, (label, group_res) in zip(tabs, results.items()):
    with tab:
        any_r = next(iter(group_res.values()))
        x_labels = any_r["table"]["Digit"].astype(str)
        title_suffix = " vs ".join(group_res.keys())

        fig = go.Figure()
        for color, (g, r) in zip(COLORS, group_res.items()):
            fig.add_bar(x=x_labels, y=r["table"]["Actual"], name=f"{g} actual", marker_color=color)
        fig.add_scatter(x=x_labels, y=any_r["table"]["Benford"], name="Benford expected",
                        mode="lines+markers", line=dict(color="#E45756", width=3))
        fig.update_layout(title=f"{label}: {title_suffix} vs Benford ({target})", barmode="group",
                          xaxis_title="Digit", yaxis_title="Proportion",
                          yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        cdf_fig = go.Figure()
        for color, (g, r) in zip(COLORS, group_res.items()):
            cdf_fig.add_scatter(x=x_labels, y=r["a_cdf"], name=f"{g} cumulative", mode="lines",
                                line=dict(color=color))
        cdf_fig.add_scatter(x=x_labels, y=any_r["b_cdf"], name="Benford cumulative", mode="lines",
                            line=dict(color="#E45756", dash="dash"))
        cdf_fig.update_layout(title=f"{label}: Cumulative Distributions (KS test)",
                              xaxis_title="Digit", yaxis_title="Cumulative proportion",
                              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(cdf_fig, width="stretch")

        # Difference from Benford by period
        diff_fig = go.Figure()
        for color, (g, r) in zip(COLORS, group_res.items()):
            diff_fig.add_bar(x=x_labels, y=r["table"]["Difference"], name=g, marker_color=color)
        diff_fig.update_layout(title=f"{label}: Actual minus Benford by Period", barmode="group",
                               xaxis_title="Digit", yaxis_title="Difference", yaxis_tickformat="+.1%",
                               legend=dict(orientation="h", y=1.1))
        st.plotly_chart(diff_fig, width="stretch")
        figures[label] = (fig, cdf_fig, diff_fig)

        st.markdown("#### Statistical Results")
        for g, r in group_res.items():
            different = r["ks"] > r["crit"]
            st.markdown(f"**{g} vs Benford**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Observations (n)", f"{r['n']:,}")
            c2.metric("KS Statistic", f"{r['ks']:.4f}")
            c3.metric(f"Critical Value (alpha={alpha})", f"{r['crit']:.4f}")
            st.write(f"KS Statistic: {r['ks']:.4f}")
            st.write(f"Critical Value: {r['crit']:.4f}")
            if different:
                st.error(f"{g}: KS Statistic > Critical Value, so the distribution IS statistically "
                         "different from Benford's Law (reject the null hypothesis).")
            else:
                st.success(f"{g}: KS Statistic <= Critical Value, so the distribution is NOT statistically "
                           "different from Benford's Law (fail to reject the null hypothesis).")
            r["conclusion"] = "Different from Benford" if different else "Consistent with Benford"

        if label in between:
            ks_ab, crit_ab = between[label]
            ga, gb = list(group_res.keys())
            st.markdown(f"**{ga} vs {gb} (two-sample KS)**")
            st.write(f"KS Statistic: {ks_ab:.4f}")
            st.write(f"Critical Value: {crit_ab:.4f}")
            if ks_ab > crit_ab:
                st.error(f"The digit distributions for {ga} and {gb} ARE statistically different.")
            else:
                st.success(f"The digit distributions for {ga} and {gb} are NOT statistically different.")

        # Combined digit table
        comb = pd.DataFrame({"Digit": any_r["table"]["Digit"], "Benford": any_r["table"]["Benford"]})
        for g, r in group_res.items():
            comb[f"{g} Count"] = r["table"]["Count"].values
            comb[f"{g} Actual"] = r["table"]["Actual"].values
            comb[f"{g} Difference"] = r["table"]["Difference"].values
        for g in group_res:
            group_res[g]["combined"] = comb
        fmt_cols = {c: "{:.4f}" for c in comb.columns if c not in ("Digit",) and not c.endswith("Count")}
        st.dataframe(comb.style.format(fmt_cols), width="stretch")

rows = []
for label, group_res in results.items():
    for g, r in group_res.items():
        rows.append({"Test": label, "Comparison": f"{g} vs Benford", "Column": target, "n": r["n"],
                     "KS Statistic": round(r["ks"], 6), "Critical Value": round(r["crit"], 6),
                     "Alpha": alpha, "Conclusion": r["conclusion"]})
    if label in between:
        ga, gb = list(group_res.keys())
        ks_ab, crit_ab = between[label]
        rows.append({"Test": label, "Comparison": f"{ga} vs {gb}", "Column": target,
                     "n": group_res[ga]["n"] + group_res[gb]["n"],
                     "KS Statistic": round(ks_ab, 6), "Critical Value": round(crit_ab, 6), "Alpha": alpha,
                     "Conclusion": "Periods differ" if ks_ab > crit_ab else "Periods not different"})
ks_summary = pd.DataFrame(rows)
st.subheader("KS Test Summary")
st.dataframe(ks_summary, width="stretch")

# ---------------------------------------------------------------
# 6. Downloads
# ---------------------------------------------------------------
st.subheader("Download & Share")
c1, c2, c3, c4 = st.columns(4)

c1.download_button("Summary statistics (CSV)", summary.to_csv().encode(),
                   f"{target}_summary_stats.csv", "text/csv")
c2.download_button("KS results (CSV)", ks_summary.to_csv(index=False).encode(),
                   f"{target}_ks_results.csv", "text/csv")
c3.download_button("Filtered data (CSV)", filtered.to_csv(index=False).encode(),
                   "filtered_data.csv", "text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary Stats")
    ks_summary.to_excel(writer, sheet_name="KS Results", index=False)
    for label, group_res in results.items():
        next(iter(group_res.values()))["combined"].to_excel(writer, sheet_name=label[:31], index=False)
c4.download_button("Full report (Excel)", excel_buffer.getvalue(), f"{target}_benford_report.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Shareable HTML report with interactive charts
html_parts = [
    f"<html><head><meta charset='utf-8'><title>Benford Report - {target}</title></head><body>",
    f"<h1>Benford's Law Report: {target}</h1>",
    f"<p>Groups compared: {', '.join(groups.keys())}</p>",
    "<h2>Summary Statistics</h2>", summary.to_html(float_format="{:,.2f}".format),
    "<h2>KS Test Results</h2>", ks_summary.to_html(index=False),
]
first = True
for label, figs in figures.items():
    html_parts.append(f"<h2>{label}</h2>")
    for f in figs:
        html_parts.append(f.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        first = False
    html_parts.append(next(iter(results[label].values()))["combined"]
                      .to_html(index=False, float_format="{:.4f}".format))
html_parts.append("</body></html>")
st.download_button("Shareable HTML report (charts + stats)", "\n".join(html_parts).encode(),
                   f"{target}_benford_report.html", "text/html")