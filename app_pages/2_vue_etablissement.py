import streamlit as st
import pandas as pd
import sys, os
# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import explicite recommandé (évite les collisions de noms)
from utils.functions import *

# -------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# =========================================================
# PARAMÈTRES (mais sans dépendances implicites dans functions.py)
# =========================================================

# Exclusions DNB (final)
DNB_EPREUVE_EXCLUDE = {
    "DICTÉE", "DICTEE",
    "RÉDACTION", "REDACTION",
    "GRAMMAIRE ET COMPRÉHENSION", "GRAMMAIRE ET COMPREHENSION",
}

# =========================================================
# GUARD + CHARGEMENT DATA
# =========================================================
required_cols = {"session", "etablissement", "examen", "bloc_epreuve", "epreuve", "moyenne", "operateur"}
if "df" not in st.session_state:
    st.error("Aucune donnée chargée dans st.session_state['df'].")
    st.stop()

raw_df = st.session_state["df"].copy()
missing = required_cols - set(raw_df.columns)
if missing:
    st.error(f"Colonnes manquantes dans df: {sorted(missing)}")
    st.stop()

# Nettoyage standardisé (inclut session->int + normalisation strings de base)
df = clean_df(raw_df)

# Sessions dynamiques depuis le df (et filtrage cohérent)
SESSIONS = sorted(df["session"].dropna().unique().tolist())
df_all = df[df["session"].isin(SESSIONS)].copy()

# Années dérivées des sessions (pour les KPIs/deltas)
YEAR_CUR = max(SESSIONS) if SESSIONS else 2025
YEAR_PREV = sorted(SESSIONS)[-2] if len(SESSIONS) >= 2 else YEAR_CUR

# Colonnes normalisées (créées une seule fois)
df_all["examen_u"] = norm_series(df_all["examen"])
df_all["bloc_u"] = norm_series(df_all["bloc_epreuve"])
df_all["epreuve_u"] = norm_series(df_all["epreuve"])
df_all["operateur_u"] = norm_series(df_all["operateur"])

# BAC : 1 ligne par (session, etablissement)
bac_net_global = (
    df_all[df_all["examen_u"] == "BAC"]
    .groupby(["session", "etablissement"], as_index=False)
    .agg(moyenne=("moyenne", "mean"))
)

# DNB FINAL : 1 ligne par (session, etablissement)
dnb_net_global = (
    df_all[(df_all["examen_u"] == "DNB") & (df_all["bloc_u"] == "DNB_FINAL")]
    .groupby(["session", "etablissement"], as_index=False)
    .agg(moyenne=("moyenne", "mean"))
)


# =========================================================
# SIDEBAR: ÉTABLISSEMENT (OSUI)
# =========================================================
st.sidebar.header("Sélection")

df_osui = df_all[df_all["operateur_u"] == "OSUI"].copy()
etabs = sorted(df_osui["etablissement"].dropna().unique().tolist())
if not etabs:
    st.error("Aucun établissement OSUI trouvé dans les données filtrées.")
    st.stop()

etab = st.sidebar.selectbox("Établissement", etabs)
df_e = df_all[df_all["etablissement"] == etab].copy()

# =========================================================
# TITRE + KPIs
# =========================================================
st.title(f"Établissement : {etab}")
st.subheader("Indicateurs clés")

# BAC rank global
bac_net_global = df_all[df_all["examen_u"] == "BAC"].copy()
rank_bac_global, n_bac = build_rank_pivot_with_total(bac_net_global, group_col=None)

# DNB rank global (DNB_FINAL)
dnb_net_global = df_all[(df_all["examen_u"] == "DNB") & (df_all["bloc_u"] == "DNB_FINAL")].copy()
rank_dnb_global, n_dnb = build_rank_pivot_with_total(dnb_net_global, group_col=None)

c1, c2 = st.columns(2)

with c1:
    s_bac = series_by_sessions(df_e[df_e["examen_u"] == "BAC"], sessions=SESSIONS)
    v, dlt = current_and_delta(s_bac, year_cur=YEAR_CUR, year_prev=YEAR_PREV)

    st.metric(
        f"Moyenne BAC ({YEAR_CUR})",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta="—" if pd.isna(dlt) else f"{dlt:+.2f}",
        border=True,
    )
    st.caption("Rang")
    st.dataframe(
        get_rank_row_over_total(rank_bac_global, n_bac, etab=etab, sessions=SESSIONS),
        hide_index=True,
        width='stretch',
        height=70,
    )

with c2:
    dnb_mask = (df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")
    s_dnb = series_by_sessions(df_e[dnb_mask], sessions=SESSIONS)
    v, dlt = current_and_delta(s_dnb, year_cur=YEAR_CUR, year_prev=YEAR_PREV)

    st.metric(
        f"Moyenne DNB (FINAL) ({YEAR_CUR})",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta="—" if pd.isna(dlt) else f"{dlt:+.2f}",
        border=True,
    )
    st.caption("Rang")
    st.dataframe(
        get_rank_row_over_total(rank_dnb_global, n_dnb, etab=etab, sessions=SESSIONS),
        hide_index=True,
        width='stretch',
        height=70,
    )

st.markdown("### Classement réseau")

years = sorted(df_all["session"].dropna().astype(int).unique().tolist())



tab1, tab2 = st.tabs(["BAC", "DNB"])

# Un seul toggle pour piloter toute la page
mode_swarm = st.toggle("Vue par proximité", value=False)

# if mode_swarm:
#     st.caption("Chaque point est un établissement :  plus les points sont serrés verticalement, plus les résultats entre les écoles sont similaires.")


with tab1:
    display_comparison_row(bac_net_global, "BAC", etab, mode_swarm)
with tab2:
    display_comparison_row(dnb_net_global, "DNB", etab, mode_swarm)

# # Application au DNB
# display_comparison_row(dnb_net_global, "DNB", etab, mode_swarm)

st.divider()

# =========================================================
# ECART AU RESEAU (delta)
# =========================================================
st.subheader("Écart aux moyennes du réseau")

delta_long = build_delta_long(
    df_all=df_all,
    df_etab=df_e,
    sessions=SESSIONS,
    dnb_exclude={x.upper().strip() for x in DNB_EPREUVE_EXCLUDE},  # cohérent avec epreuve_u
)

bar_input = delta_long_to_bar_input(delta_long)
fig = make_bac_dnb_bar(
    df=bar_input,
    sessions=SESSIONS,
    dnb_epreuve_exclude=DNB_EPREUVE_EXCLUDE,  # ok même si bar_input a peu de colonnes DNB (sécurisé)
    dnb_color_offset=5,
)
st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)



st.divider()

# =========================================================
# ECART AU RESEAU (delta)
# =========================================================
st.subheader("Evolution des moyennes par épreuve")
# =========================================================
# TABS
# =========================================================
tab_bac, tab_dnb, tab_spe = st.tabs(["BAC — Épreuves finales", "DNB — Épreuves finales", "BAC — Spécialités"])

# =========================================================
# TAB 1 — BAC Épreuves finales (cartes par bloc_epreuve)
# =========================================================
with tab_bac:
    bac_etab = df_e[df_e["examen_u"] == "BAC"].copy()
    bac_net = df_all[df_all["examen_u"] == "BAC"].copy()

    rank_pivot_bac_bloc, n_bac_bloc = build_rank_pivot_with_total(
        bac_net.dropna(subset=["bloc_epreuve"]).copy(),
        group_col="bloc_epreuve",
    )

    target_blocs = [
        ("EDS", {"EDS"}),
        ("EAF", {"EAF"}),
        ("Grand Oral", {"GO", "GRAND ORAL"}),
        ("Philosophie", {"PHILO", "PHILOSOPHIE"}),
    ]

    card_items = []
    for label, accepted in target_blocs:
        m = bac_etab["bloc_u"].isin(accepted)
        if bac_etab[m].empty:
            continue

        s = series_by_sessions(bac_etab[m], sessions=SESSIONS)

        # Valeur pivot (bloc_epreuve) la plus fréquente pour ce label
        bloc_value = bac_etab.loc[m, "bloc_epreuve"].mode().iloc[0]
        rank_df = get_rank_row_over_total(
            rank_pivot=rank_pivot_bac_bloc,
            n_by_session=n_bac_bloc,
            etab=etab,
            sessions=SESSIONS,
            group_col="bloc_epreuve",
            group_value=bloc_value,
        )
        card_items.append((label, s, rank_df))

    render_cards_grid(
        card_items=card_items,
        sessions=SESSIONS,
        year_current=YEAR_CUR,
        year_prev=YEAR_PREV,
        cols=4,
    )

# =========================================================
# TAB 2 — DNB Épreuves finales (cartes par epreuve)
# =========================================================
with tab_dnb:
    dnb_etab = df_e[(df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")].copy()
    dnb_net = df_all[(df_all["examen_u"] == "DNB") & (df_all["bloc_u"] == "DNB_FINAL")].copy()

    dnb_etab = dnb_etab[~dnb_etab["epreuve_u"].isin({x.upper().strip() for x in DNB_EPREUVE_EXCLUDE})].copy()
    dnb_net = dnb_net[~dnb_net["epreuve_u"].isin({x.upper().strip() for x in DNB_EPREUVE_EXCLUDE})].copy()

    if not dnb_etab.empty:
        rank_pivot_dnb, n_dnb_by_ep = build_rank_pivot_with_total(
            dnb_net.dropna(subset=["epreuve"]).copy(),
            group_col="epreuve",
        )

        card_items = []
        for epr in sorted(dnb_etab["epreuve"].dropna().unique().tolist()):
            s = series_by_sessions(dnb_etab[dnb_etab["epreuve"] == epr], sessions=SESSIONS)
            rank_df = get_rank_row_over_total(
                rank_pivot=rank_pivot_dnb,
                n_by_session=n_dnb_by_ep,
                etab=etab,
                sessions=SESSIONS,
                group_col="epreuve",
                group_value=epr,
            )
            card_items.append((str(epr), s, rank_df))

        render_cards_grid(
            card_items=card_items,
            sessions=SESSIONS,
            year_current=YEAR_CUR,
            year_prev=YEAR_PREV,
            cols=5,
        )

# =========================================================
# TAB 3 — BAC Spécialités (EDS) (cartes par epreuve)
# =========================================================
with tab_spe:
    spe_etab = df_e[(df_e["examen_u"] == "BAC") & (df_e["bloc_u"] == "EDS")].copy()
    spe_net = df_all[(df_all["examen_u"] == "BAC") & (df_all["bloc_u"] == "EDS")].copy()

    if not spe_etab.empty:
        rank_pivot_spe, n_spe = build_rank_pivot_with_total(
            spe_net.dropna(subset=["epreuve"]).copy(),
            group_col="epreuve",
        )

        card_items = []
        for epr in sorted(spe_etab["epreuve"].dropna().unique().tolist()):
            s = series_by_sessions(spe_etab[spe_etab["epreuve"] == epr], sessions=SESSIONS)
            rank_df = get_rank_row_over_total(
                rank_pivot=rank_pivot_spe,
                n_by_session=n_spe,
                etab=etab,
                sessions=SESSIONS,
                group_col="epreuve",
                group_value=epr,
            )
            card_items.append((str(epr), s, rank_df))

        render_cards_grid(
            card_items=card_items,
            sessions=SESSIONS,
            year_current=YEAR_CUR,
            year_prev=YEAR_PREV,
            cols=4,
        )
