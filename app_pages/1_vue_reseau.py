import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

G10 = px.colors.qualitative.G10


# ============================
# HELPERS
# ============================

def clean_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """Standardise types/strings. (No session filtering here.)"""
    df = df_in.copy()

    # Types
    df["session"] = df["session"].astype(int)

    # String normalization
    str_cols = ["operateur", "examen", "bloc_epreuve", "type_epreuve", "epreuve"]
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Canonical casing
    df["examen"] = df["examen"].str.upper()
    df["epreuve"] = df["epreuve"].str.upper()

    return df


def mean_by_year(df_in: pd.DataFrame, value_col: str = "moyenne") -> pd.Series:
    return df_in.groupby("session")[value_col].mean().sort_index()


def metric_pack(
    series_by_year: pd.Series,
    sessions: list[int],
    year_current: int = 2025,
    year_prev: int = 2024,
):
    """Return (current_value, delta_vs_prev, chart_df_indexed_by_session)."""
    values = {y: series_by_year.get(y, pd.NA) for y in sessions}
    values = {y: float(v) if pd.notna(v) else float("nan") for y, v in values.items()}

    v_cur = values.get(year_current, float("nan"))
    v_prev = values.get(year_prev, float("nan"))
    delta = None if (pd.isna(v_cur) or pd.isna(v_prev)) else round(v_cur - v_prev, 2)

    chart = (
        pd.DataFrame({"session": sessions, "mean": [values[y] for y in sessions]})
        .set_index("session")
    )
    return v_cur, delta, chart


def compute_indicator_metrics(df: pd.DataFrame, sessions: list[int]) -> dict:
    metrics = {}

    bac_series = mean_by_year(df[df["examen"] == "BAC"])
    metrics["BAC"] = metric_pack(bac_series, sessions=sessions)

    dnb_mask = (df["examen"] == "DNB") & (df["bloc_epreuve"] == "DNB_FINAL")
    dnb_series = mean_by_year(df[dnb_mask])
    metrics["DNB"] = metric_pack(dnb_series, sessions=sessions)

    return metrics


def build_trend(df: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat(
        [
            df[df["examen"] == "BAC"].assign(indicateur="BAC"),
            df[df["examen"] == "DNB"].assign(indicateur="DNB"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    return (
        df_all.groupby(["session", "operateur", "indicateur"], dropna=False)["moyenne"]
        .mean()
        .reset_index(name="mean")
    )


def make_trend_figure(trend: pd.DataFrame):
    fig = px.line(
        trend,
        x="session",
        y="mean",
        color="operateur",
        facet_col="indicateur",
        markers=True,
        color_discrete_sequence=G10,
    )

    fig.update_xaxes(type="category", title_text="")
    fig.update_yaxes(title_text="", showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_layout(
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=0, b=0),
    )

    return fig


# ============================
# MAIN
# ============================

raw_df = st.session_state["df"]
df = clean_df(raw_df)

# sessions dynamiques depuis le df
SESSIONS = sorted(df["session"].dropna().unique().tolist())

# (optionnel) si vous voulez limiter aux 3 dernières sessions :
# SESSIONS = SESSIONS[-3:]

# filtrage cohérent
df = df[df["session"].isin(SESSIONS)].copy()

nb_etablissements = df["etablissement"].nunique()
metrics = compute_indicator_metrics(df, sessions=SESSIONS)

bac_v25, bac_delta, _ = metrics["BAC"]
dnb_v25, dnb_delta, _ = metrics["DNB"]

st.subheader("Indicateurs clés")

c1, c2 = st.columns([1, 3])

with c1:
    st.metric("Nombre d’établissements", nb_etablissements, border=True)

    st.metric(
        "Moyenne BAC (2025)",
        value="—" if pd.isna(bac_v25) else f"{bac_v25:.2f}",
        delta=None if bac_delta is None else f"{bac_delta:+.2f}",
        border=True,
    )

    st.metric(
        "Moyenne DNB (2025)",
        value="—" if pd.isna(dnb_v25) else f"{dnb_v25:.2f}",
        delta=None if dnb_delta is None else f"{dnb_delta:+.2f}",
        border=True,
    )

with c2:
    trend = build_trend(df)
    fig_trend = make_trend_figure(trend)
    st.plotly_chart(fig_trend, use_container_width=True)


# ============================
# BAR CHART : BAC par bloc_epreuve x session (groupé)
# ============================
st.divider()

st.write("### Évolution des moyennes par épreuve)")

# bac = df[df["examen"] == "BAC"].copy()

# bar_df = (
#     bac.groupby(["bloc_epreuve", "session"], dropna=False)["moyenne"]
#     .mean()
#     .reset_index()
# )

# bar_df = bar_df[bar_df["bloc_epreuve"].notna()].copy()

# # session en catégorie (string) pour Plotly + ordre stable
# bar_df["session"] = bar_df["session"].astype(int)
# bar_df = bar_df[bar_df["session"].isin(SESSIONS)]
# bar_df["session"] = bar_df["session"].astype(str)

# fig_bar = px.bar(
#     bar_df,
#     x="bloc_epreuve",
#     y="moyenne",
#     color="session",
#     barmode="group",
#     category_orders={"session": [str(s) for s in SESSIONS]},
#     text_auto=".2f",
#     color_discrete_sequence=G10,
# )

# fig_bar.update_layout(
#     bargap=0.35,
#     bargroupgap=0.0,
#     xaxis_title="Bloc d’épreuve (BAC)",
#     yaxis_title="Moyenne",
#     legend_title_text="Session",
#     margin=dict(l=20, r=20, t=30, b=20),
# )
# fig_bar.update_xaxes(title_text="")
# fig_bar.update_yaxes(title_text="")

# fig_bar.update_layout(
#     legend_title_text="",
#     legend=dict(
#         orientation="h",
#         yanchor="top",
#         y=1.15,
#         xanchor="center",
#         x=0.5,
#     ),
#     margin=dict(l=0, r=0, t=0, b=0),
# )

# st.plotly_chart(fig_bar, use_container_width=True)

import plotly.express as px
import pandas as pd

G10 = px.colors.qualitative.G10

# ----------------------------
# PARAMÈTRES
# ----------------------------
DNB_EPREUVE_EXCLUDE = {
    "GRAMMAIRE ET COMPRÉHENSION",
    "DICTÉE",
    "RÉDACTION",
}



DNB_COLOR_OFFSET = 5  # décalage dans G10 pour distinguer visuellement BAC / DNB


def make_bac_dnb_bar(df: pd.DataFrame, sessions: list[int]):
    dfx = df.copy()

    # Sécurisation casse / espaces
    dfx["bloc_epreuve_u"] = dfx["bloc_epreuve"].astype(str).str.upper().str.strip()
    dfx["epreuve_u"] = dfx["epreuve"].astype(str).str.upper().str.strip()

    # ============================
    # BAC → blocs d’épreuves
    # ============================
    bac = dfx[dfx["examen"] == "BAC"].copy()

    bac_bar = (
        bac.groupby(["bloc_epreuve", "session"], dropna=False)["moyenne"]
           .mean()
           .reset_index()
    )

    bac_bar = bac_bar[bac_bar["bloc_epreuve"].notna()].copy()
    bac_bar["indicateur"] = "BAC"
    bac_bar["group"] = bac_bar["bloc_epreuve"]

    # ============================
    # DNB → épreuves (avec exclusions)
    # ============================
    dnb = dfx[
        (dfx["examen"] == "DNB")
        & (~dfx["epreuve_u"].isin(DNB_EPREUVE_EXCLUDE))
    ].copy()

    dnb_bar = (
        dnb.groupby(["epreuve", "session"], dropna=False)["moyenne"]
           .mean()
           .reset_index()
    )

    dnb_bar = dnb_bar[dnb_bar["epreuve"].notna()].copy()
    dnb_bar["indicateur"] = "DNB"
    dnb_bar["group"] = dnb_bar["epreuve"]


    # ============================
    # Combinaison
    # ============================
    bar_df = pd.concat([bac_bar, dnb_bar], ignore_index=True)

    bar_df["session"] = bar_df["session"].astype(int)
    bar_df = bar_df[bar_df["session"].isin(sessions)].copy()

    bar_df["session_str"] = bar_df["session"].astype(str)

    # Libellé X final
    bar_df["x_label"] = bar_df["group"].astype(str)

    # Série couleur (indicateur + session)
    bar_df["serie"] = bar_df["indicateur"] + " " + bar_df["session_str"]

    # Ordre X : BAC puis DNB
    x_order = (
        sorted(bar_df.loc[bar_df["indicateur"] == "BAC", "x_label"].unique())
        + sorted(bar_df.loc[bar_df["indicateur"] == "DNB", "x_label"].unique())
    )

    # ============================
    # Couleurs (G10 séparées)
    # ============================
    color_map = {}

    for i, s in enumerate(sessions):
        color_map[f"BAC {s}"] = G10[i % len(G10)]
        color_map[f"DNB {s}"] = G10[(DNB_COLOR_OFFSET + i) % len(G10)]

    # ============================
    # Figure
    # ============================
    fig = px.bar(
        bar_df,
        x="x_label",
        y="moyenne",
        color="serie",
        barmode="group",
        category_orders={"x_label": x_order},
        color_discrete_map=color_map,
        text_auto=".2f",
    )

    fig.update_layout(
        xaxis_title="Blocs / Épreuves (BAC + DNB)",
        yaxis_title="Moyenne",
        legend_title_text="",
        bargap=0.01,
        bargroupgap=0.05,
        width=900,
        margin=dict(l=00, r=00, t=20, b=20),
        legend=dict(
        orientation="h",
        yanchor="top",
        y=1.15,
        xanchor="center",
        x=0.5,
        ),
    )
    fig.update_traces(textfont_size=50, textangle=0, textposition="outside")

    fig.update_xaxes(tickangle=-25,title="")
    fig.update_yaxes(tickangle=-25,title="")

    return fig


# ============================
# USAGE
# ============================
fig = make_bac_dnb_bar(df, sessions=SESSIONS)
st.plotly_chart(fig, use_container_width=True)


BAC_BLOC_EXCLUDE = {
    "GO",
    "GRAND ORAL",
    "PHILO",
    "PHILOSOPHIE",
    "FRANÇAIS",
    "FRANCAIS",
}

df_bac = df[df["examen"] == "BAC"].copy()

df_bac["bloc_u"] = (
    df_bac["epreuve"]
    .astype(str)
    .str.upper()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df_bac = df_bac[~df_bac["bloc_u"].isin(BAC_BLOC_EXCLUDE)].copy()
df_bac = df_bac[df_bac["epreuve"].notna()].copy()

# Moyennes par bloc x session
agg = (
    df_bac.groupby(["epreuve", "session"], dropna=False)["moyenne"]
          .mean()
          .reset_index(name="mean")
)

# Pivot => colonnes moy_<année>
wide_bac = agg.pivot(index="epreuve", columns="session", values="mean").reset_index()

# Renommer sessions -> moy_YYYY
wide_bac = wide_bac.rename(columns={y: f"moy_{y}" for y in SESSIONS})

# Ranks (1 = meilleure moyenne)
for y in SESSIONS:
    col = f"moy_{y}"
    if col in wide_bac.columns:
        wide_bac[f"rank_{y}"] = wide_bac[col].rank(ascending=False, method="min")


moy_cols = [f"moy_{y}" for y in SESSIONS]

def row_to_list(row):
    return [None if pd.isna(row[c]) else float(row[c]) for c in moy_cols]

wide_bac["trend_bar"]  = wide_bac.apply(row_to_list, axis=1)


# Évol% première -> dernière session
y0, y1 = SESSIONS[0], SESSIONS[-1]
wide_bac["evol_pct"] = (wide_bac[f"moy_{y1}"] / wide_bac[f"moy_{y0}"] - 1.0) * 100

# Score 0..100 (clamp)
clip_min, clip_max = -10.0, 10.0
evol_clipped = wide_bac["evol_pct"].clip(clip_min, clip_max)
wide_bac["evol_progress"] = ((evol_clipped - clip_min) / (clip_max - clip_min) * 100)
wide_bac["evol_progress"] = wide_bac["evol_progress"].fillna(0).round(0)



# bornes communes (évite autoscale)
y_min = float(np.nanmin(wide_bac[moy_cols].values))
y_max = float(np.nanmax(wide_bac[moy_cols].values))

# optionnel : tri
wide_bac = wide_bac.sort_values("evol_pct", ascending=False)

column_config = {
    "epreuve": st.column_config.TextColumn("BAC - EDS", width="medium"),
}

for y in SESSIONS:
    column_config[f"moy_{y}"]  = st.column_config.NumberColumn(f"Moy {y}", format="%.2f")
    column_config[f"rank_{y}"] = st.column_config.NumberColumn(f"Rank {y}", format="%.0f")

column_config.update({
    "evol_pct": st.column_config.NumberColumn(f"Évol {SESSIONS[0]}→{SESSIONS[-1]}", format="%+.1f%%"),
    # "trend_line": st.column_config.LineChartColumn("Tendance (ligne)", y_min=y_min, y_max=y_max),
    "trend_bar": st.column_config.BarChartColumn("Tendance (barres)", y_min=y_min, y_max=y_max),
    # "evol_progress": st.column_config.ProgressColumn(
    #     "Évol (progress)",
    #     min_value=0,
    #     max_value=100,
    #     format="%.0f",  # <-- clé pour éviter 6300% / 10000%
    #     help=f"Score 0–100 basé sur une évolution clampée entre {clip_min}% et {clip_max}%",
    # ),
})
ordered_cols = (
    ["epreuve"]
    + [f"moy_{y}" for y in SESSIONS]
    + [f"rank_{y}" for y in SESSIONS]
    + ["evol_pct", "trend_bar"]
)

wide_bac = wide_bac[ordered_cols]

st.dataframe(
    wide_bac,
    hide_index=True,
    use_container_width=True,
    column_config=column_config,
)
