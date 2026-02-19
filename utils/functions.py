from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px

# Optionnel: seulement si vous gardez les fonctions UI dans ce fichier
import streamlit as st
import altair as alt
import collections
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Palette Plotly
G10 = px.colors.qualitative.G10

# Ajoute ici ta config de téléchargement
PLOTLY_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'scale': 2 # Haute qualité
    },
    'displaylogo': False
}


# =========================================================
# CLEANING / BASE
# =========================================================
def clean_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """Standardise types/strings. (No session filtering here.)"""
    df = df_in.copy()

    # Types
    if "session" in df.columns:
        df["session"] = df["session"].astype(int)

    # String normalization
    str_cols = ["operateur", "examen", "bloc_epreuve", "type_epreuve", "epreuve"]
    for col in str_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # Canonical casing
    if "examen" in df.columns:
        df["examen"] = df["examen"].str.upper()
    if "epreuve" in df.columns:
        df["epreuve"] = df["epreuve"].str.upper()

    return df


def norm_series(s: pd.Series) -> pd.Series:
    """Normalise une série texte: uppercase + strip + espaces multiples."""
    return (
        s.astype(str)
         .str.upper()
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )


# =========================================================
# KPI / SERIES
# =========================================================
def mean_by_year(df_in: pd.DataFrame, value_col: str = "moyenne") -> pd.Series:
    """Mean(value_col) par session."""
    if df_in.empty:
        return pd.Series(dtype="float64")
    return df_in.groupby("session")[value_col].mean().sort_index()


def metric_pack(
    series_by_year: pd.Series,
    sessions: list[int],
    year_current: int,
    year_prev: int,
) -> tuple[float, float | None, pd.DataFrame]:
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


def compute_indicator_metrics(
    df: pd.DataFrame,
    sessions: list[int],
    year_current: int,
    year_prev: int,
    dnb_final_bloc: str = "DNB_FINAL",
) -> dict:
    """
    Calcule les KPIs principaux (BAC, DNB final).
    Retourne un dict: {"BAC": (v_cur, delta, chart_df), "DNB": (...)}
    """
    metrics: dict = {}

    bac_series = mean_by_year(df[df["examen"] == "BAC"])
    metrics["BAC"] = metric_pack(bac_series, sessions=sessions, year_current=year_current, year_prev=year_prev)

    dnb_mask = (df["examen"] == "DNB") & (df["bloc_epreuve"] == dnb_final_bloc)
    dnb_series = mean_by_year(df[dnb_mask])
    metrics["DNB"] = metric_pack(dnb_series, sessions=sessions, year_current=year_current, year_prev=year_prev)

    return metrics


def series_by_sessions(
    dfi: pd.DataFrame,
    sessions: list[int],
    value_col: str = "moyenne",
) -> pd.Series:
    """Série mean(value_col) par session sur `sessions`. NaN si absent."""
    if dfi.empty:
        return pd.Series(index=sessions, dtype="float64")

    s = dfi.groupby("session")[value_col].mean()
    return pd.Series({y: float(s.get(y, np.nan)) for y in sessions})


def current_and_delta(
    s: pd.Series,
    year_cur: int,
    year_prev: int,
) -> tuple[float, float]:
    """(valeur year_cur, delta year_cur vs year_prev)."""
    v_cur = s.get(year_cur, np.nan)
    v_prev = s.get(year_prev, np.nan)
    delta = np.nan if (pd.isna(v_cur) or pd.isna(v_prev)) else float(v_cur - v_prev)
    return v_cur, delta


def _safe_wavg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy()
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_numpy()
    mask = (~np.isnan(v)) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def build_trend_means_by_operator(df: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat(
        [
            df[df["examen"] == "BAC"].assign(indicateur="BAC"),
            df[df["examen"] == "DNB"].assign(indicateur="DNB"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    df_all["moyenne"] = pd.to_numeric(df_all["moyenne"], errors="coerce")
    df_all["nb_presents"] = pd.to_numeric(df_all["nb_presents"], errors="coerce").fillna(0.0)

    out = (
        df_all.groupby(["session", "operateur", "indicateur"], dropna=False)
        .apply(lambda x: pd.Series({"mean": _safe_wavg(x["moyenne"], x["nb_presents"])}))
        .reset_index()
    )
    return out


def build_trend_sigma_network(df: pd.DataFrame) -> pd.DataFrame:
    df_all = pd.concat(
        [
            df[df["examen"] == "BAC"].assign(indicateur="BAC"),
            df[df["examen"] == "DNB"].assign(indicateur="DNB"),
        ],
        ignore_index=True,
    ).drop_duplicates()

    df_all["ecart_type"] = pd.to_numeric(df_all["ecart_type"], errors="coerce")
    df_all["nb_presents"] = pd.to_numeric(df_all["nb_presents"], errors="coerce").fillna(0.0)

    out = (
        df_all.groupby(["session", "indicateur"], dropna=False)
        .apply(lambda x: pd.Series({"sigma": _safe_wavg(x["ecart_type"], x["nb_presents"])}))
        .reset_index()
    )
    return out


def make_trend_means_with_sigma_subplot(
    trend_means: pd.DataFrame,
    trend_sigma: pd.DataFrame,
):
    """
    2x2 subplot:
      - row 1: mean par opérateur (lines)
      - row 2: sigma réseau (area)
    Cols: BAC, DNB (selon trend_means/trend_sigma)
    """
    # ordre stable
    indicateurs = [x for x in ["BAC", "DNB"] if x in set(trend_means["indicateur"]) or x in set(trend_sigma["indicateur"])]
    if not indicateurs:
        indicateurs = sorted(set(trend_means["indicateur"]).union(set(trend_sigma["indicateur"])))

    fig = make_subplots(
        rows=2,
        cols=len(indicateurs),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.8, 0.2],
        subplot_titles=indicateurs,
    )

    # couleurs opérateur cohérentes
    ops = sorted(trend_means["operateur"].dropna().unique().tolist())
    color_map = {op: G10[i % len(G10)] for i, op in enumerate(ops)}

    for c, indic in enumerate(indicateurs, start=1):
        # --- row 1: moyennes opérateur ---
        sub_m = trend_means[trend_means["indicateur"] == indic].sort_values("session")
        for op in ops:
            d = sub_m[sub_m["operateur"] == op]
            if d.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["session"],
                    y=d["mean"],
                    mode="lines+markers",
                    name=op,
                    showlegend=(c == 1),
                    legendgroup=op,
                    line=dict(width=3, color=color_map[op]),
                    marker=dict(size=7),

                ),
                row=1,
                col=c,
            )

        # --- row 2: sigma réseau (area) ---
        sub_s = (
            trend_sigma[trend_sigma["indicateur"] == indic]
            .dropna(subset=["sigma"])
            .sort_values("session")
        )
        if not sub_s.empty:

            # --- area (sans hover) ---
            fig.add_trace(
                go.Scatter(
                    x=sub_s["session"],
                    y=sub_s["sigma"],
                    mode="none",
                    fill="tozeroy",
                    fillcolor="rgba(120,120,120,0.25)",
                    name="",  # 👈 évite "trace 3"
                    text="Evolution de la dispersion 23→25",
                    hoveron="fills",
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=c,
            )

            # --- ligne + points + valeurs ---
            fig.add_trace(
                go.Scatter(
                    x=sub_s["session"],
                    y=sub_s["sigma"],
                    mode="lines+markers+text",
                    text=[f"{v:.2f}" for v in sub_s["sigma"]],
                    textposition="top center",
                    line=dict(color="rgba(120,120,120,1)", width=1),
                    marker=dict(size=7),
                    # hovertemplate="Evolution de la dispersion 23→25<extra></extra>",
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=c,
            )


        # axes
        fig.update_xaxes(type="category", title_text="", row=2, col=c)
        fig.update_yaxes(title_text="", row=1, col=c)
        fig.update_yaxes(title_text="σ", row=2, col=c,range=[3.4, 3.8])

    fig.update_layout(
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=20, r=20, t=0, b=0),
        height=450,
    )

    return fig






# =========================================================
# RANKS (réseau)
# =========================================================
def build_rank_pivot_with_total(df_scope: pd.DataFrame, group_col: str | None = None):
    """
    Rang (1 = meilleur) sur mean(moyenne) par session.
    - group_col=None  -> rang global par établissement et session
    - group_col="..." -> rang par (groupe, établissement) et session
    Retourne (rank_pivot, n_by_session)
    """
    if df_scope.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")

    group_keys = ["session", "etablissement"] if group_col is None else [group_col, "session", "etablissement"]
    tmp = (
        df_scope.groupby(group_keys, dropna=False)["moyenne"]
                .mean()
                .reset_index(name="mean")
    )

    rank_group = ["session"] if group_col is None else [group_col, "session"]
    tmp["rank"] = tmp.groupby(rank_group)["mean"].rank(ascending=False, method="min")

    pivot_index = "etablissement" if group_col is None else [group_col, "etablissement"]
    rank_pivot = tmp.pivot_table(index=pivot_index, columns="session", values="rank", aggfunc="first")

    n_by_session = tmp.groupby("session")["etablissement"].nunique()
    return rank_pivot, n_by_session


def format_rank_over_total(rank_val, total_val) -> str:
    if pd.isna(rank_val) or pd.isna(total_val) or float(total_val) <= 0:
        return "—"
    return f"{int(round(float(rank_val))):d}/{int(total_val):d}"


def get_rank_row_over_total(
    rank_pivot: pd.DataFrame,
    n_by_session: pd.Series,
    etab: str,
    sessions: list[int],
    group_col: str | None = None,
    group_value: str | None = None,
) -> pd.DataFrame:
    """
    Retourne une DF 1 ligne: colonnes = sessions (str) au format 'rang/total'.
    """
    def empty_row():
        return pd.DataFrame([{str(y): "—" for y in sessions}])

    if rank_pivot is None or rank_pivot.empty:
        return empty_row()

    if group_col is None:
        if etab not in rank_pivot.index:
            return empty_row()
        s = rank_pivot.loc[etab]
    else:
        if group_value is None:
            return empty_row()
        key = (group_value, etab)
        if key not in rank_pivot.index:
            return empty_row()
        s = rank_pivot.loc[key]

    return pd.DataFrame([{
        str(y): format_rank_over_total(s.get(y, np.nan), n_by_session.get(y, np.nan))
        for y in sessions
    }])

def get_rank_row_top_percent(
    rank_pivot: pd.DataFrame,
    n_by_session: pd.Series,
    etab: str,
    sessions: list[int],
    group_col: str | None = None,
    group_value: str | None = None,
) -> pd.DataFrame:
    """
    Retourne une DF 1 ligne: colonnes = sessions (str) au format 'Top xx%'.
    Plus petit = meilleur (Top 1% = excellent).
    """

    def empty_row():
        return pd.DataFrame([{str(y): "—" for y in sessions}])

    if rank_pivot is None or rank_pivot.empty:
        return empty_row()

    # --- récupération série des rangs
    if group_col is None:
        if etab not in rank_pivot.index:
            return empty_row()
        s = rank_pivot.loc[etab]
    else:
        if group_value is None:
            return empty_row()
        key = (group_value, etab)
        if key not in rank_pivot.index:
            return empty_row()
        s = rank_pivot.loc[key]

    row = {}
    for y in sessions:
        r = s.get(y, np.nan)
        n = n_by_session.get(y, np.nan)

        if pd.isna(r) or pd.isna(n) or int(n) == 0:
            row[str(y)] = "—"
        else:
            r_i = int(r)
            n_i = int(n)
            top_pct = 100.0 * r_i / n_i
            row[str(y)] = f"{top_pct:.0f}%"

    return pd.DataFrame([row])



# =========================================================
#UI CARDS (Streamlit + Altair)
# =========================================================

# -------------------------
# STD helper (dataset déjà agrégé, 1 ligne / session après filtrage)
# -------------------------
def std_df_by_sessions(
    df: pd.DataFrame,
    sessions: list[int],
    std_col: str = "ecart_type",
    fallback_col: str | None = "ecart_type_imputed",
) -> pd.DataFrame:
    """
    Retourne DF: session(str), std(float)
    Fallback: si std NaN ou 0 => fallback_col (si dispo).
    """
    tmp = df.copy()

    cols = ["session", std_col]
    if fallback_col and fallback_col in tmp.columns:
        cols.append(fallback_col)
    else:
        fallback_col = None

    tmp = tmp[cols].drop_duplicates(subset=["session"])

    if fallback_col:
        std = tmp[std_col].copy()
        fb = tmp[fallback_col]
        std = std.where(~std.isna() & (std != 0), fb)
        tmp["std"] = std
    else:
        tmp["std"] = tmp[std_col]

    out = tmp[tmp["session"].isin(sessions)][["session", "std"]].copy()
    out = out.dropna(subset=["std"])
    out["session"] = out["session"].astype(int).astype(str)
    return out.sort_values("session")


def _center_std(std_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    std_df: ["session"(str), "std"(float)]
    => ajoute std_centered = std - mean(std), zero=0
    """
    if std_df is None or std_df.empty:
        return std_df
    tmp = std_df.copy()
    mu = float(tmp["std"].mean())
    tmp["std_centered"] = tmp["std"] - mu
    tmp["zero"] = 0.0
    return tmp


# -------------------------
# CARD
# -------------------------

def render_card(
    title: str,
    s: pd.Series,
    rank_df: pd.DataFrame,
    sessions: list[int],
    year_current: int,
    year_prev: int,
    y_domain=(9, 18),
    std_df: pd.DataFrame | None = None,   # NEW: DF avec colonnes ["session","std"]
    std_domain=(2, 6),                    # optionnel
):
    """Carte KPI + mini courbe moyenne + mini courbe écart-type + mini tableau rang/total."""
    if s.isna().all():
        return

    v_cur = s.get(year_current, np.nan)
    v_prev = s.get(year_prev, np.nan)

    value_str = "—" if pd.isna(v_cur) else f"{float(v_cur):.2f}"
    delta_str = "—" if (pd.isna(v_cur) or pd.isna(v_prev)) else f"{float(v_cur - v_prev):+.2f}"

    with st.container(border=True):
        st.metric(title, value=value_str, delta=delta_str)

        # -------------------------
        # 1) Mini courbe MOYENNE (identique à ton code)
        # -------------------------
        chart_df = pd.DataFrame(
            {"session": [str(y) for y in sessions], "mean": [s.get(y, np.nan) for y in sessions]}
        ).dropna(subset=["mean"])

        if not chart_df.empty:
            chart = (
                alt.Chart(chart_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("session:N", title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("mean:Q", title=None, scale=alt.Scale(domain=list(y_domain))),
                )
                .properties(height=120)
            )
            st.caption("Moyenne")
            st.altair_chart(chart, width="stretch")

        # -------------------------
        # 2) Mini courbe ECART-TYPE (simple, même style, sans axe Y)
        # std_df doit être: session(str/int), std(float)
        # -------------------------
        if std_df is not None and not std_df.empty:
            tmp = std_df.copy()
            tmp = tmp.dropna(subset=["std"]).copy()
            tmp["session"] = tmp["session"].astype(int).astype(str)
            tmp = tmp[tmp["session"].isin([str(y) for y in sessions])].sort_values("session")

            if not tmp.empty:
                chart_std = (
                    alt.Chart(tmp)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("session:N", title=None, axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y("std:Q", title=None, scale=alt.Scale(domain=list(std_domain))),
                        tooltip=[
                            alt.Tooltip("session:N", title="Session"),
                            alt.Tooltip("std:Q", title="Écart-type", format=".2f"),
                        ],
                    )
                    .properties(height=120)
                )
                st.caption("Ecart-type")
                st.altair_chart(chart_std, width="stretch")

        st.caption("Top x % (rang/total)")
        st.dataframe(rank_df, hide_index=True, width="stretch", height=70)

# -------------------------
# GRID
# -------------------------
def render_cards_grid(
    card_items: list[tuple[str, pd.Series, pd.DataFrame, pd.DataFrame | None]],
    sessions: list[int],
    year_current: int,
    year_prev: int,
    cols: int = 3,
):
    card_items = [(t, s, r, std) for (t, s, r, std) in card_items if not s.isna().all()]
    if not card_items:
        return

    for i in range(0, len(card_items), cols):
        row = st.columns(cols)
        for j in range(cols):
            if i + j >= len(card_items):
                break
            title, s, rank_df, std_df = card_items[i + j]
            with row[j]:
                render_card(
                    title=title,
                    s=s,
                    rank_df=rank_df,
                    sessions=sessions,
                    year_current=year_current,
                    year_prev=year_prev,
                    std_df=std_df,
                )



# =========================================================
# DELTA établissement vs réseau
# =========================================================
def agg_group_session_mean(dfi: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Agrège mean(moyenne) par (group_col, session) -> [group_col, session, mean]."""
    if dfi.empty:
        return pd.DataFrame(columns=[group_col, "session", "mean"])
    return (
        dfi.groupby([group_col, "session"], dropna=False)["moyenne"]
           .mean()
           .reset_index(name="mean")
    )


def build_delta_long_one_exam(
    df_all: pd.DataFrame,
    df_etab: pd.DataFrame,
    exam: str,
    group_col: str,
    sessions: list[int],
    exclude_values: set[str] | None = None,
) -> pd.DataFrame:
    """
    Retourne un DF long: [exam, group_label, session, delta]
    delta = mean(etablissement) - mean(reseau)
    """
    d_all = df_all[(df_all["examen_u"] == exam) & (df_all["session"].isin(sessions))].copy()
    d_etb = df_etab[(df_etab["examen_u"] == exam) & (df_etab["session"].isin(sessions))].copy()

    if exclude_values is not None:
        col_u = "bloc_u" if group_col == "bloc_epreuve" else "epreuve_u"
        d_all = d_all[~d_all[col_u].isin(exclude_values)].copy()
        d_etb = d_etb[~d_etb[col_u].isin(exclude_values)].copy()

    d_all = d_all[d_all[group_col].notna()].copy()
    d_etb = d_etb[d_etb[group_col].notna()].copy()

    net = agg_group_session_mean(d_all, group_col=group_col).rename(columns={"mean": "mean_net"})
    etb = agg_group_session_mean(d_etb, group_col=group_col).rename(columns={"mean": "mean_etab"})

    if net.empty and etb.empty:
        return pd.DataFrame(columns=["exam", "group_label", "session", "delta"])

    merged = pd.merge(etb, net, on=[group_col, "session"], how="outer")
    merged["delta"] = merged["mean_etab"] - merged["mean_net"]
    merged["exam"] = exam
    merged = merged.rename(columns={group_col: "group_label"})

    return merged[["exam", "group_label", "session", "delta"]].dropna(subset=["delta"])


def build_delta_long(
    df_all: pd.DataFrame,
    df_etab: pd.DataFrame,
    sessions: list[int],
    dnb_exclude: set[str] | None = None,
) -> pd.DataFrame:
    """Concat BAC (bloc_epreuve) + DNB (epreuve) en delta long."""
    parts = []

    if (df_etab["examen_u"] == "BAC").any():
        parts.append(
            build_delta_long_one_exam(
                df_all=df_all,
                df_etab=df_etab,
                exam="BAC",
                group_col="bloc_epreuve",
                sessions=sessions,
                exclude_values=None,
            )
        )

    if (df_etab["examen_u"] == "DNB").any():
        parts.append(
            build_delta_long_one_exam(
                df_all=df_all,
                df_etab=df_etab,
                exam="DNB",
                group_col="epreuve",
                sessions=sessions,
                exclude_values=dnb_exclude,
            )
        )

    if not parts:
        return pd.DataFrame(columns=["exam", "group_label", "session", "delta"])

    return pd.concat(parts, ignore_index=True)


def delta_long_to_bar_input(delta_long: pd.DataFrame) -> pd.DataFrame:
    """
    Adaptateur vers le format attendu par make_bac_dnb_bar :
    colonnes: [examen, bloc_epreuve, epreuve, session, moyenne]
    où moyenne = delta.
    """
    d = delta_long.copy()
    d["examen"] = d["exam"]
    d["moyenne"] = d["delta"]
    d["bloc_epreuve"] = np.where(d["examen"] == "BAC", d["group_label"], np.nan)
    d["epreuve"] = np.where(d["examen"] == "DNB", d["group_label"], np.nan)
    d["session"] = d["session"].astype(int)
    return d[["examen", "bloc_epreuve", "epreuve", "session", "moyenne"]]


# =========================================================
# GRAPHIQUE (bar BAC + DNB)
# =========================================================
def make_bac_dnb_bar(
    df: pd.DataFrame,
    sessions: list[int],
    dnb_epreuve_exclude: set[str] | None = None,
    dnb_color_offset: int = 5,
):
    """
    Barres groupées: moyennes (ou ici deltas) par bloc/épreuve et session.
    """
    dfx = df.copy()

    # Normalisation (utile pour exclusions DNB)
    dfx["bloc_u"] = norm_series(dfx["bloc_epreuve"]) if "bloc_epreuve" in dfx.columns else ""
    dfx["epreuve_u"] = norm_series(dfx["epreuve"]) if "epreuve" in dfx.columns else ""

    dnb_epreuve_exclude_u = {x.upper().strip() for x in dnb_epreuve_exclude} if dnb_epreuve_exclude else set()

    # BAC -> bloc_epreuve
    bac = dfx[dfx["examen"] == "BAC"].copy()
    bac_bar = (
        bac.groupby(["bloc_epreuve", "session"], dropna=False)["moyenne"]
           .mean()
           .reset_index()
    )
    bac_bar = bac_bar[bac_bar["bloc_epreuve"].notna()].copy()
    bac_bar["indicateur"] = "BAC"
    bac_bar["group"] = bac_bar["bloc_epreuve"]

    # DNB -> epreuve, exclusions
    dnb = dfx[(dfx["examen"] == "DNB") & (~dfx["epreuve_u"].isin(dnb_epreuve_exclude_u))].copy()
    dnb_bar = (
        dnb.groupby(["epreuve", "session"], dropna=False)["moyenne"]
           .mean()
           .reset_index()
    )
    dnb_bar = dnb_bar[dnb_bar["epreuve"].notna()].copy()
    dnb_bar["indicateur"] = "DNB"
    dnb_bar["group"] = dnb_bar["epreuve"]

    bar_df = pd.concat([bac_bar, dnb_bar], ignore_index=True)
    bar_df["session"] = bar_df["session"].astype(int)
    bar_df = bar_df[bar_df["session"].isin(sessions)].copy()
    bar_df["session_str"] = bar_df["session"].astype(str)

    bar_df["x_label"] = bar_df["group"].astype(str)
    bar_df["serie"] = bar_df["indicateur"] + " " + bar_df["session_str"]

    x_order = (
        sorted(bar_df.loc[bar_df["indicateur"] == "BAC", "x_label"].unique())
        + sorted(bar_df.loc[bar_df["indicateur"] == "DNB", "x_label"].unique())
    )

    # Couleurs: BAC sur G10[i], DNB sur G10[offset+i]
    color_map = {}
    for i, s in enumerate(sessions):
        color_map[f"BAC {s}"] = G10[i % len(G10)]
        color_map[f"DNB {s}"] = G10[(dnb_color_offset + i) % len(G10)]

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
        yaxis_title="Moyenne (ou Écart vs réseau)",
        legend_title_text="",
        bargap=0.01,
        bargroupgap=0.05,
        margin=dict(l=0, r=0, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5,
        ),
        height=520,
    )

    fig.update_traces(textfont_size=12, textangle=0, textposition="outside")
    fig.update_xaxes(tickangle=-25, title="")
    fig.update_yaxes(title="")

    return fig



def get_swarm_fig(df_year, title, etab, point_size=13, fig_width=350):
    """
    Génère une figure Beeswarm (algorithme de précision) avec :
    - Une zone d'excellence (Top 25% du réseau)
    - Un badge dynamique indiquant le Top % de l'établissement sélectionné
    - Une ligne verticale de repère
    """
    # 1. Nettoyage et tri des données (fondamental pour l'algorithme de collision)
    df_plot = df_year.dropna(subset=["moyenne"]).sort_values("moyenne").copy()
    if df_plot.empty:
        return None

    # 2. Calculs statistiques pour le positionnement
    total_etabs = len(df_plot)
    # On calcule le seuil du Top 25% (quartile supérieur)
    threshold_top_25 = df_plot["moyenne"].quantile(0.75)

    min_x, max_x = df_plot["moyenne"].min(), df_plot["moyenne"].max()
    # Sécurité pour l'échelle si toutes les notes sont identiques
    if min_x == max_x:
        min_x -= 1
        max_x += 1

    # 3. Algorithme de collision Swarm (Positionnement des points en C-shape)
    bin_fraction = 0.95
    gap_multiplier = 1.2
    bin_counter = collections.Counter()
    list_of_rows = []

    for _, row in df_plot.iterrows():
        x_val = row["moyenne"]
        # Attribution d'un "bin" (colonne verticale) basé sur la largeur de la figure
        bin_idx = (((fig_width * bin_fraction * (x_val - min_x)) / (max_x - min_x)) // point_size)
        bin_counter.update([bin_idx])
        list_of_rows.append({
            "etablissement": row["etablissement"],
            "x": x_val,
            "y_slot": bin_counter[bin_idx],
            "bin": bin_idx
        })

    # Calcul des offsets pour éviter le chevauchement entre bins adjacents
    current_bin, offset = -1, 0
    for i, row in enumerate(list_of_rows):
        if current_bin != row["bin"]:
            current_bin, offset = row["bin"], 0
        for j in range(i):
            other = list_of_rows[j]
            if other["bin"] == current_bin - 1:
                while (other["y_slot"] == row["y_slot"] + offset and
                       (((fig_width * (row["x"] - other["x"])) / (max_x - min_x)) // point_size) < 1):
                    offset += 1
        row["y_slot"] += offset
        # Centrage vertical autour de zéro
        sign = -1 if row["y_slot"] % 2 == 1 else 1
        row["y"] = (row["y_slot"] // 2) * sign * point_size * gap_multiplier

    # Ajustement pour aligner parfaitement les nombres pairs
    for row in list_of_rows:
        if bin_counter[row["bin"]] % 2 == 0:
            row["y"] -= (point_size * gap_multiplier) / 2

    df_swarm = pd.DataFrame(list_of_rows)
    df_swarm["highlight"] = np.where(df_swarm["etablissement"] == etab, "Sélection", "Autres")

    # Rang réel (1 = meilleur)
    df_plot["rang"] = df_plot["moyenne"].rank(ascending=False, method="min").astype(int)

    df_swarm = df_swarm.merge(
        df_plot[["etablissement", "rang"]],
        on="etablissement",
        how="left"
    )

    # 4. Création du graphique de base
    fig = px.scatter(
        df_swarm,
        x="x",
        y="y",
        color="highlight",
        custom_data=["rang"],
        hover_name="etablissement",
        hover_data={'highlight':False},
        color_discrete_map={"Sélection": "#FF4B4B", "Autres": "#B0B0B0"},
        title=title
    )

    # 5. Ajout de la Zone d'Excellence (Rectangle vert transparent)
    fig.add_vrect(
        x0=threshold_top_25, x1=max_x + 0.5,
        fillcolor="rgba(0, 200, 100, 0.1)", # Vert très pâle
        layer="below",
        line_width=0,
        annotation_text="Top 25%",
        annotation_position="top left",
        annotation_font=dict(size=9, color="green", style="italic")
    )

    # 6. Indicateur de l'établissement sélectionné (Ligne + Badge Top %)
    if etab in df_swarm["etablissement"].values:
        # Calcul du rang réel (1 = premier)
        df_plot["rang"] = df_plot["moyenne"].rank(ascending=False, method="min")
        rang_etab = df_plot.loc[df_plot["etablissement"] == etab, "rang"].iloc[0]
        v_x = df_plot.loc[df_plot["etablissement"] == etab, "moyenne"].iloc[0]

        # Calcul du percentile
        top_pct = (rang_etab / total_etabs) * 100

        # Ligne de repère verticale
        fig.add_vline(x=v_x, line_dash="dash", line_color="#FF4B4B", opacity=0.8)

        # Badge dynamique
        fig.add_annotation(
            x=v_x,
            y=max(df_swarm["y"]) * 1.4, # Placé au dessus de l'essaim
            text=f"<b>{etab}</b><br>TOP {top_pct:.0f}%",
            showarrow=False,
            font=dict(color="white", size=10),
            bgcolor="#FF4B4B",
            borderpad=6,
            yshift=15
        )

    # 7. Mise en forme finale
    fig.update_traces(
        marker=dict(size=point_size, line=dict(width=0.5, color='white')),
        hovertemplate="<b>%{hovertext}</b><br>Moyenne: %{x:.2f}"
    )
    fig.update_traces(
        selector=dict(name="Autres"),
        hovertemplate=(
            "Moyenne: %{x:.2f}<br>"
            "Rang: %{customdata[0]}<extra></extra>"
        )
    )


    fig.update_traces(
        selector=dict(name="Sélection"),
        hovertemplate=(
            f"<b>{etab}</b><br>"
            "Moyenne: %{x:.2f}<br>"
            "Rang: %{customdata[0]}<extra></extra>"
        )
    )


    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.1)",
        title=""
    )

    fig.update_xaxes(
        title="Moyenne BAC",
        range=[min_x - 0.5, max_x + 0.5],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.05)"
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        plot_bgcolor="white"
    )

    return fig


def display_comparison_row(df_global, label_title, etab_name, mode_swarm_active):
    """
    Affiche une ligne de 3 colonnes pour un examen donné (BAC ou DNB)
    """
    # Récupération des 3 dernières années disponibles pour ce DataFrame spécifique
    years_available = sorted(df_global["session"].dropna().astype(int).unique().tolist())
    target_years = years_available[-3:]

    cols = st.columns(3)

    for i, year in enumerate(target_years):
        with cols[i]:
            # Filtrage pour l'année
            df_year = df_global[df_global["session"].astype(int) == year].copy()

            # On s'assure d'avoir une seule ligne par établissement (moyenne déjà calculée normalement)
            df_year = df_year.groupby(["etablissement"], as_index=False).agg(moyenne=("moyenne", "mean"))

            if mode_swarm_active:
                # Appel de la fonction Swarm que nous avons finalisée
                fig = get_swarm_fig(df_year, f"{label_title} {year}", etab=etab_name)
                if fig:
                    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG, key=f"swarm_{label_title}_{year}")
            else:
                # Mode Barres (Décroissant)
                df_year = df_year.sort_values("moyenne", ascending=False)
                df_year["rang"] = np.arange(1, len(df_year) + 1)  # 1 = meilleur
                df_year["highlight"] = np.where(df_year["etablissement"] == etab_name, "Sélection", "Autres")


                fig = px.bar(
                    df_year,
                    x="etablissement",
                    y="moyenne",
                    color="highlight",
                    category_orders={"etablissement": df_year["etablissement"].tolist()},
                    color_discrete_map={"Sélection": "#FF4B4B", "Autres": "#B0B0B0"},
                    title=f"{label_title} {year}",
                    custom_data=["rang", "etablissement"],
                )
                # Sécurité : l'établissement existe ?
                if etab_name in df_year["etablissement"].values:
                    rang_etab = int(df_year.loc[df_year["etablissement"] == etab_name, "rang"].iloc[0])
                    total_etabs = len(df_year)

                    # Ligne de repère (sur la catégorie X)
                    fig.add_vline(
                        x=etab_name,
                        line_dash="dash",
                        line_color="#FF4B4B",
                        opacity=0.8
                    )

                    # Badge au-dessus (coordonnées en "paper" pour Y)
                    fig.add_annotation(
                        x=etab_name,
                        xref="x",
                        y=1.08,
                        yref="paper",
                        text=f"<b>{etab_name}</b><br>{rang_etab:.0f}/{total_etabs}",
                        showarrow=False,
                        font=dict(color="white", size=10),
                        bgcolor="#FF4B4B",
                        borderpad=6
                    )

                fig.update_xaxes(showticklabels=False, title="")
                fig.update_yaxes(title="")
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=350)
                # Autres: pas de nom, mais moyenne + rang
                fig.update_traces(
                    selector=dict(name="Autres"),
                    hovertemplate="Moyenne: %{y:.2f}<br>Rang: %{customdata[0]}<extra></extra>"
                )

                # Sélection: nom + moyenne + rang
                fig.update_traces(
                    selector=dict(name="Sélection"),
                    hovertemplate="<b>%{customdata[1]}</b><br>Moyenne: %{y:.2f}<br>Rang: %{customdata[0]}<extra></extra>"
                )

                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG, key=f"bar_{label_title}_{year}")






#######

def _wavg(v: pd.Series, w: pd.Series) -> float:
    v = pd.to_numeric(v, errors="coerce").to_numpy()
    w = pd.to_numeric(w, errors="coerce").fillna(0.0).to_numpy()
    mask = (~np.isnan(v)) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))

def aggregate_etab_mean_sigma(df_year: pd.DataFrame) -> pd.DataFrame:
    """
    Input: lignes par épreuve (ou déjà agrégées)
    Output: 1 ligne par établissement avec:
      - mean (μ) : moyenne pondérée par nb_presents
      - sigma (σ) : écart-type pondéré par nb_presents
    """
    d = df_year.copy()
    # sécurité colonnes
    for c in ["moyenne", "ecart_type", "nb_presents"]:
        if c not in d.columns:
            raise KeyError(f"Colonne manquante: {c}")

    d["moyenne"] = pd.to_numeric(d["moyenne"], errors="coerce")
    d["ecart_type"] = pd.to_numeric(d["ecart_type"], errors="coerce")
    d["nb_presents"] = pd.to_numeric(d["nb_presents"], errors="coerce").fillna(0.0)

    out = (
        d.groupby("etablissement", as_index=False)
         .apply(lambda x: pd.Series({
             "mean": _wavg(x["moyenne"], x["nb_presents"]),
             "sigma": _wavg(x["ecart_type"], x["nb_presents"]),
             "n": float(x["nb_presents"].sum()),
         }))
         .reset_index()
    )
    # groupby+apply crée parfois une colonne index inutile
    if "level_0" in out.columns:
        out = out.drop(columns=["level_0"])
    out = out.dropna(subset=["mean", "sigma"])
    return out

def compute_top_percent(df_etab: pd.DataFrame, etab_name: str) -> tuple[int | None, int | None, float | None]:
    """
    Top% sur la MOYENNE (plus haut = meilleur).
    Retourne (rang, total, top_pct).
      - rang: 1 = meilleur
      - top_pct: ex 0.28 => Top 28%
    """
    if df_etab.empty or etab_name not in df_etab["etablissement"].values:
        return None, None, None

    tmp = df_etab.sort_values("mean", ascending=False).reset_index(drop=True)
    tmp["rang"] = np.arange(1, len(tmp) + 1)
    total = len(tmp)
    rang = int(tmp.loc[tmp["etablissement"] == etab_name, "rang"].iloc[0])
    top_pct = rang / total  # 0.28 => top 28%
    return rang, total, top_pct


def quadrant_label(mean: float, sigma: float, mean_ref: float, sigma_ref: float) -> tuple[str, str]:
    """
    Retourne (label, couleur) selon la position par rapport aux références.
    mean_ref / sigma_ref = lignes de séparation (cadran).
    """
    up_mean = mean >= mean_ref
    up_sigma = sigma >= sigma_ref

    # Convention: Y=mean, X=sigma
    if up_mean and (not up_sigma):
        return ("Moyenne ↑ / Dispersion ↓", "rgba(46, 204, 113, 0.18)")  # vert (souhaitable)
    if up_mean and up_sigma:
        return ("Moyenne ↑ / Dispersion ↑", "rgba(241, 196, 15, 0.18)")  # jaune
    if (not up_mean) and (not up_sigma):
        return ("Moyenne ↓ / Dispersion ↓", "rgba(52, 152, 219, 0.18)")  # bleu
    return ("Moyenne ↓ / Dispersion ↑", "rgba(231, 76, 60, 0.18)")       # rouge

def get_quadrant_scatter_fig(
    df_etab: pd.DataFrame,
    title: str,
    etab: str,
    x_label: str = "Écart-type (σ)",
    y_label: str = "Moyenne (μ)",
    split: str = "median",          # "median" recommandé
    show_other_names_in_hover: bool = False,
):
    """
    df_etab attendu: colonnes ['etablissement', 'mean', 'sigma'] (1 ligne / etab)
    Rendu:
      - Quadrants colorés (4 zones)
      - Nuage gris des autres
      - Point rouge + label (quadrant)
      - Lignes de cadran (verticale + horizontale)
    """
    if df_etab is None or df_etab.empty:
        return None

    d = df_etab.copy().dropna(subset=["mean", "sigma"])
    if d.empty:
        return None

    # split (référence cadran) sur l'année
    if split == "mean":
        sigma_ref = float(d["sigma"].mean())
        mean_ref = float(d["mean"].mean())
    else:
        sigma_ref = float(d["sigma"].median())
        mean_ref = float(d["mean"].median())

    # bornes (padding)
    x_min, x_max = float(d["sigma"].min()), float(d["sigma"].max())
    y_min, y_max = float(d["mean"].min()), float(d["mean"].max())
    pad_x = 0.12 * (x_max - x_min) if x_max > x_min else 0.5
    pad_y = 0.18 * (y_max - y_min) if y_max > y_min else 0.5
    x0, x1 = x_min - pad_x, x_max + pad_x
    y0, y1 = y_min - pad_y, y_max + pad_y

    # séparer sélection / autres
    me = d[d["etablissement"] == etab]
    other = d[d["etablissement"] != etab]

    fig = go.Figure()

    # quadrants (fond léger)
    quad_colors = {
        "HL": "rgba(46, 204, 113, 0.10)",  # mean↑ sigma↓
        "HR": "rgba(241, 196, 15, 0.10)",  # mean↑ sigma↑
        "BL": "rgba(52, 152, 219, 0.10)",  # mean↓ sigma↓
        "BR": "rgba(231, 76, 60, 0.10)",   # mean↓ sigma↑
    }
    # BL
    fig.add_shape(type="rect", x0=x0, x1=sigma_ref, y0=y0, y1=mean_ref,
                  fillcolor=quad_colors["BL"], line=dict(width=0))
    # BR
    fig.add_shape(type="rect", x0=sigma_ref, x1=x1, y0=y0, y1=mean_ref,
                  fillcolor=quad_colors["BR"], line=dict(width=0))
    # HL
    fig.add_shape(type="rect", x0=x0, x1=sigma_ref, y0=mean_ref, y1=y1,
                  fillcolor=quad_colors["HL"], line=dict(width=0))
    # HR
    fig.add_shape(type="rect", x0=sigma_ref, x1=x1, y0=mean_ref, y1=y1,
                  fillcolor=quad_colors["HR"], line=dict(width=0))

    # lignes cadran
    fig.add_shape(type="line", x0=sigma_ref, x1=sigma_ref, y0=y0, y1=y1,
                  line=dict(width=1, dash="dot", color="rgba(120,120,120,0.55)"))
    fig.add_shape(type="line", x0=x0, x1=x1, y0=mean_ref, y1=mean_ref,
                  line=dict(width=1, dash="dot", color="rgba(120,120,120,0.55)"))

    # autres établissements (gris)
    if show_other_names_in_hover:
        hover_other = "<b>%{customdata}</b><br>μ=%{y:.2f}<br>σ=%{x:.2f}<extra></extra>"
        custom = other["etablissement"]
    else:
        hover_other = "μ=%{y:.2f}<br>σ=%{x:.2f}<extra></extra>"
        custom = None

    fig.add_trace(
        go.Scatter(
            x=other["sigma"],
            y=other["mean"],
            mode="markers",
            marker=dict(size=12, color="rgba(150,150,150,0.60)"),
            showlegend=False,
            hovertemplate=hover_other,
            customdata=custom,
        )
    )

    # établissement sélectionné : point rouge + ligne verticale rouge + badge quadrant
    if not me.empty:
        x_me = float(me["sigma"].iloc[0])
        y_me = float(me["mean"].iloc[0])

        # ligne verticale rouge sur sigma de l'établissement (comme ton visuel)
        fig.add_shape(
            type="line",
            x0=x_me, x1=x_me, y0=y0, y1=y1,
            line=dict(width=3, dash="dash", color="#FF4B4B"),
            opacity=0.85,
        )

        # point rouge
        fig.add_trace(
            go.Scatter(
                x=[x_me],
                y=[y_me],
                mode="markers",
                marker=dict(size=18, color="#FF4B4B", line=dict(width=1, color="white")),
                showlegend=False,
                hovertemplate=f"<b>{etab}</b><br>μ=%{{y:.2f}}<br>σ=%{{x:.2f}}<extra></extra>",
            )
        )

        # badge = nom + quadrant
        q_label, _ = quadrant_label(y_me, x_me, mean_ref, sigma_ref)
        badge = f"<b>{etab}</b><br>{q_label}"

        fig.add_annotation(
            x=x_me, y=y_me,
            xref="x", yref="y",
            text=badge,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF4B4B",
            ax=0, ay=-75,
            font=dict(color="white", size=12),
            bgcolor="#FF4B4B",
            borderpad=8,
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
    )
    fig.update_xaxes(range=[x0, x1], title_text=x_label, zeroline=False)
    fig.update_yaxes(range=[y0, y1], title_text=y_label, zeroline=False)

    return fig

def display_mean_sigma_row(df_global, label_title, etab_name):
    """
    3 colonnes (3 dernières années) : Mean (Y) vs σ (X) avec quadrants + highlight.
    df_global = df filtré sur examen (BAC ou DNB).
    """
    years_available = sorted(df_global["session"].dropna().astype(int).unique().tolist())
    target_years = years_available[-3:]
    cols = st.columns(3)

    for i, year in enumerate(target_years):
        with cols[i]:
            df_year = df_global[df_global["session"].astype(int) == year].copy()

            # agrégation 1 ligne / etab => mean + sigma
            df_etab = aggregate_etab_mean_sigma(df_year)

            fig = get_quadrant_scatter_fig(
                df_etab=df_etab,
                title=f"{label_title} {year}",
                etab=etab_name,
                x_label="Écart-type",
                y_label=f"Moyenne {label_title}",
                split="median",
                show_other_names_in_hover=False,
            )

            if fig:
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=f"quad_{label_title}_{year}")
