import streamlit as st
import pandas as pd
import numpy as np
import sys, os

# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import explicite recommandé (évite les collisions de noms)
from utils.functions import *

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

# Années courante / précédente dérivées des sessions
YEAR_CUR = max(SESSIONS) if SESSIONS else 2025
YEAR_PREV = sorted(SESSIONS)[-2] if len(SESSIONS) >= 2 else YEAR_CUR

nb_etablissements = df["etablissement"].nunique()

metrics = compute_indicator_metrics(
    df=df,
    sessions=SESSIONS,
    year_current=YEAR_CUR,
    year_prev=YEAR_PREV,
    dnb_final_bloc="DNB_FINAL",
)

bac_v_cur, bac_delta, _ = metrics["BAC"]
dnb_v_cur, dnb_delta, _ = metrics["DNB"]

st.title(f"Réseau EFE Maroc - Vue globale")


st.subheader("Indicateurs clés")

c1, c2 = st.columns([1, 3])

with c1:
    st.metric("Nombre d’établissements", nb_etablissements, border=True)

    st.metric(
        f"Moyenne BAC ({YEAR_CUR})",
        value="—" if pd.isna(bac_v_cur) else f"{bac_v_cur:.2f}",
        delta=None if bac_delta is None else f"{bac_delta:+.2f}",
        border=True,
    )

    st.metric(
        f"Moyenne DNB ({YEAR_CUR})",
        value="—" if pd.isna(dnb_v_cur) else f"{dnb_v_cur:.2f}",
        delta=None if dnb_delta is None else f"{dnb_delta:+.2f}",
        border=True,
    )

with c2:
    trend = build_trend(df)
    fig_trend = make_trend_figure(trend)
    st.plotly_chart(fig_trend, width='stretch', config=PLOTLY_CONFIG)

# ============================
# BAR CHART : BAC + DNB par session (groupé)
# ============================
st.divider()
st.write("### Évolution des moyennes par épreuve")

# ----------------------------
# PARAMÈTRES (exclusions DNB)
# ----------------------------
DNB_EPREUVE_EXCLUDE = {
    "GRAMMAIRE ET COMPRÉHENSION",
    "DICTÉE",
    "RÉDACTION",
}

fig = make_bac_dnb_bar(
    df=df,
    sessions=SESSIONS,
    dnb_epreuve_exclude=DNB_EPREUVE_EXCLUDE,
    dnb_color_offset=5,  # optionnel
)
st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


st.divider()

# =========================================================
# ECART AU RESEAU (delta)
# =========================================================
st.subheader("Évolution des moyennes par spécialité")
# ============================
# TABLE : BAC (EDS) - pivots, ranks, tendances
# ============================

BAC_BLOC_EXCLUDE = {
    "GO",
    "GRAND ORAL",
    "PHILO",
    "PHILOSOPHIE",
    "FRANÇAIS",
    "FRANCAIS",
}

df_bac = df[df["examen"] == "BAC"].copy()

# NOTE: vous filtrez ici sur "epreuve" (comme votre code original)
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

wide_bac["trend_bar"] = wide_bac.apply(row_to_list, axis=1)

# Évol% première -> dernière session
if len(SESSIONS) >= 2:
    y0, y1 = SESSIONS[0], SESSIONS[-1]
    wide_bac["evol_pct"] = (wide_bac[f"moy_{y1}"] / wide_bac[f"moy_{y0}"] - 1.0) * 100
else:
    wide_bac["evol_pct"] = np.nan

# bornes communes (évite autoscale)
if moy_cols and wide_bac[moy_cols].to_numpy().size:
    y_min = float(np.nanmin(wide_bac[moy_cols].values))
    y_max = float(np.nanmax(wide_bac[moy_cols].values))
else:
    y_min, y_max = 0.0, 1.0

# optionnel : tri
wide_bac = wide_bac.sort_values("evol_pct", ascending=False, na_position="last")

column_config = {
    "epreuve": st.column_config.TextColumn("BAC - EDS", width="medium"),
}

for y in SESSIONS:
    column_config[f"moy_{y}"] = st.column_config.NumberColumn(f"Moy {y}", format="%.2f")
    column_config[f"rank_{y}"] = st.column_config.NumberColumn(f"Rank {y}", format="%.0f")

if len(SESSIONS) >= 2:
    column_config.update({
        "evol_pct": st.column_config.NumberColumn(f"Évol {SESSIONS[0]}→{SESSIONS[-1]}", format="%+.1f%%"),
        "trend_bar": st.column_config.BarChartColumn("Tendance (barres)", y_min=y_min, y_max=y_max),
    })
else:
    column_config.update({
        "evol_pct": st.column_config.NumberColumn("Évol", format="%+.1f%%"),
        "trend_bar": st.column_config.BarChartColumn("Tendance (barres)", y_min=y_min, y_max=y_max),
    })

ordered_cols = (
    ["epreuve"]
    + [f"moy_{y}" for y in SESSIONS]
    + [f"rank_{y}" for y in SESSIONS]
    + ["evol_pct", "trend_bar"]
)

# sécurisation: ne garder que les colonnes existantes
ordered_cols = [c for c in ordered_cols if c in wide_bac.columns]
wide_bac = wide_bac[ordered_cols]

st.dataframe(
    wide_bac,
    hide_index=True,
    width='stretch',
    column_config=column_config,
)
