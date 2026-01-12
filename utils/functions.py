from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px

# Optionnel: seulement si vous gardez les fonctions UI dans ce fichier
import streamlit as st
import altair as alt


# Palette Plotly
G10 = px.colors.qualitative.G10


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


# =========================================================
# TREND (Plotly)
# =========================================================
def build_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend opérateur: BAC + DNB (moyenne), par session / opérateur / indicateur.
    """
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


# =========================================================
# UI CARDS (Streamlit + Altair)
# =========================================================
def render_card(
    title: str,
    s: pd.Series,
    rank_df: pd.DataFrame,
    sessions: list[int],
    year_current: int,
    year_prev: int,
    y_domain=(9, 18),
):
    """Carte KPI + mini courbe + mini tableau rang/total."""
    if s.isna().all():
        return

    v_cur = s.get(year_current, np.nan)
    v_prev = s.get(year_prev, np.nan)

    value_str = "—" if pd.isna(v_cur) else f"{float(v_cur):.2f}"
    delta_str = "—" if (pd.isna(v_cur) or pd.isna(v_prev)) else f"{float(v_cur - v_prev):+.2f}"

    with st.container(border=True):
        st.metric(title, value=value_str, delta=delta_str)

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
            st.altair_chart(chart, width='stretch')

        st.caption("Rang")
        st.dataframe(rank_df, hide_index=True, width='stretch', height=70)


def render_cards_grid(
    card_items: list[tuple[str, pd.Series, pd.DataFrame]],
    sessions: list[int],
    year_current: int,
    year_prev: int,
    cols: int = 3,
):
    card_items = [(t, s, r) for (t, s, r) in card_items if not s.isna().all()]
    if not card_items:
        return

    for i in range(0, len(card_items), cols):
        row = st.columns(cols)
        for j in range(cols):
            if i + j >= len(card_items):
                break
            title, s, rank_df = card_items[i + j]
            with row[j]:
                render_card(
                    title=title,
                    s=s,
                    rank_df=rank_df,
                    sessions=sessions,
                    year_current=year_current,
                    year_prev=year_prev,
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
