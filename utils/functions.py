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
            # area
            fig.add_trace(
                go.Scatter(
                    x=sub_s["session"],
                    y=sub_s["sigma"],
                    mode="none",
                    name="σ réseau",
                    showlegend=False,
                    line=dict(width=2),
                    fill="tozeroy",
                    opacity=0.25,
                    fillcolor="rgba(120,120,120,0.25)",  # 👈 gris clair transparent
                    # line=dict(color="rgba(120,120,120,1)"),
                    hovertemplate="σ réseau: %{y:.2f}<extra></extra>",
                ),
                row=2,
                col=c,
            )
            # # petite ligne par-dessus pour précision
            # fig.add_trace(
            #     go.Scatter(
            #         x=sub_s["session"],
            #         y=sub_s["sigma"],
            #         mode="lines+markers",
            #         name="σ réseau (ligne)",
            #         showlegend=False,
            #         line=dict(width=2, dash="dot"),
            #         marker=dict(size=6),
            #         opacity=0.8,
            #         hovertemplate="σ réseau: %{y:.2f}<extra></extra>",
            #     ),
            #     row=2,
            #     col=c,
            #)

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
        height=480,
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
#UI CARDS (Streamlit + Altair)
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

# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go

# # Colonnes candidates pour l'écart-type (adapte si ton nom exact diffère)
# STD_COL_CANDIDATES = ["ecart_type", "ecart-type", "std", "sigma", "ecart_type_eleves", "ecart_type_resultats"]

# def pick_std_col(df: pd.DataFrame) -> str | None:
#     cols = set(df.columns)
#     for c in STD_COL_CANDIDATES:
#         if c in cols:
#             return c
#     return None


# def agg_mean_std_by_etab_session(
#     df_scope: pd.DataFrame,
#     exam: str,
#     sessions: list[int],
#     group_col: str | None = None,      # "bloc_epreuve" ou "epreuve"
#     group_value: str | None = None,    # valeur de bloc/épreuve sélectionnée
# ) -> pd.DataFrame:
#     """
#     Sortie: 1 ligne par (session, etablissement) avec mean(moyenne) + mean(ecart_type)
#     Si group_col/group_value fournis -> filtre sur ce groupe avant agrégation.
#     """
#     if df_scope.empty:
#         return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

#     std_col = pick_std_col(df_scope)
#     if std_col is None:
#         # On préfère être explicite : tu as dit que tu as la donnée,
#         # donc si elle n'est pas là, on renvoie une DF vide (plutôt que du faux).
#         return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

#     d = df_scope[(df_scope["examen_u"] == exam) & (df_scope["session"].isin(sessions))].copy()

#     if group_col is not None and group_value is not None:
#         d = d[d[group_col].astype(str) == str(group_value)].copy()

#     d = d.dropna(subset=["moyenne", std_col, "etablissement", "session"]).copy()
#     if d.empty:
#         return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

#     out = (
#         d.groupby(["session", "etablissement"], as_index=False)
#          .agg(mean=("moyenne", "mean"), std=(std_col, "mean"))
#     )
#     out["session"] = out["session"].astype(int)
#     return out


# def add_iso_ratio_guides(fig, x_min, x_max, ratios=(6, 7, 8, 9, 10)):
#     xs = np.linspace(x_min, x_max, 80)
#     for r in ratios:
#         ys = xs / float(r)
#         fig.add_trace(
#             go.Scatter(
#                 x=xs,
#                 y=ys,
#                 mode="lines",
#                 line=dict(width=1, color="rgba(0,0,0,0.18)", dash="dot"),
#                 hoverinfo="skip",
#                 showlegend=False,
#             )
#         )
#         # petit label sur la droite
#         fig.add_annotation(
#             x=float(xs[-1]),
#             y=float(ys[-1]),
#             text=f"μ/σ={r}",
#             showarrow=False,
#             font=dict(size=10, color="rgba(0,0,0,0.45)"),
#             xanchor="left",
#             yanchor="middle",
#         )


# def make_mean_std_scatter(
#     df_scope: pd.DataFrame,
#     exam: str,
#     sessions: list[int],
#     etab_selected: str,
#     color_map: dict[str, str],
#     group_col: str | None = None,      # "bloc_epreuve" (BAC) / "epreuve" (DNB)
#     group_value: str | None = None,
#     only_one_year: int | None = None,  # si tu veux forcer une seule session
# ):
#     """
#     Scatter mean (x) vs std (y).
#     Couleur = série (exam + session) via color_map (même logique que le bar chart).
#     Point sélection = rouge (#FF4B4B).
#     """
#     sess = sessions if only_one_year is None else [int(only_one_year)]

#     df_plot = agg_mean_std_by_etab_session(
#         df_scope=df_scope,
#         exam=exam,
#         sessions=sess,
#         group_col=group_col,
#         group_value=group_value,
#     )
#     if df_plot.empty:
#         return None

#     # Rang interne (1 = meilleur) par session sur mean
#     df_plot["rank"] = df_plot.groupby("session")["mean"].rank(ascending=False, method="min").astype(int)

#     # Série pour mapping couleur
#     df_plot["serie"] = df_plot["session"].apply(lambda s: f"{exam} {int(s)}")

#     # bornes utiles
#     x_min = float(df_plot["mean"].min())
#     x_max = float(df_plot["mean"].max())
#     if x_min == x_max:
#         x_min -= 0.5
#         x_max += 0.5

#     # ajoute les “cadrans” iso-ratio directement sur le scatter
#     add_iso_ratio_guides(fig, x_min, x_max, ratios=(6, 7, 8, 9, 10))

#     # Split sélection / autres
#     df_sel = df_plot[df_plot["etablissement"] == etab_selected].copy()
#     df_oth = df_plot[df_plot["etablissement"] != etab_selected].copy()

#     fig = go.Figure()

#     # Traces "Autres" : par session (couleurs alignées bar chart)
#     for s in sorted(df_oth["session"].unique().tolist()):
#         serie = f"{exam} {int(s)}"
#         dfx = df_oth[df_oth["session"] == s]

#         fig.add_trace(
#             go.Scatter(
#                 x=dfx["mean"],
#                 y=dfx["std"],
#                 mode="markers",
#                 name=serie,
#                 marker=dict(
#                     size=10,
#                     color=color_map.get(serie, "#B0B0B0"),
#                     opacity=0.75,
#                     line=dict(width=0),
#                 ),
#                 customdata=np.stack([dfx["rank"], dfx["etablissement"], dfx["session"]], axis=-1),
#                 hovertemplate=(
#                     "Établissement: %{customdata[1]}<br>"
#                     "Session: %{customdata[2]}<br>"
#                     "Moyenne: %{x:.2f}<br>"
#                     "Écart-type: %{y:.2f}<br>"
#                     "Rang: %{customdata[0]}<extra></extra>"
#                 ),
#                 showlegend=True if len(sess) > 1 else False,
#             )
#         )

#    # Trace "Sélection" : points ronds + ligne de liaison
#     if not df_sel.empty:

#         # On trie par session pour garantir une ligne chronologique propre
#         df_sel = df_sel.sort_values("session")

#         fig.add_trace(
#             go.Scatter(
#                 x=df_sel["mean"],
#                 y=df_sel["std"],
#                 mode="lines+markers",
#                 name="Sélection",
#                 line=dict(
#                     color="#111827",      # charcoal très contrasté
#                     width=2.5,
#                 ),
#                 marker=dict(
#                     size=16,
#                     color="#111827",
#                     symbol="circle",
#                     line=dict(width=2, color="white"),
#                 ),
#                 customdata=np.stack([df_sel["rank"], df_sel["session"]], axis=-1),
#                 hovertemplate=(
#                     f"<b>{etab_selected}</b><br>"
#                     "Session: %{customdata[1]}<br>"
#                     "Moyenne: %{x:.2f}<br>"
#                     "Écart-type: %{y:.2f}<br>"
#                     "Rang: %{customdata[0]}<extra></extra>"
#                 ),
#                 showlegend=True,
#             )
#         )

#     title = "Moyenne vs Écart-type"
#     if group_col and group_value:
#         title += f" — {group_value}"

#     fig.update_layout(
#         title=title,
#         height=520,
#         margin=dict(l=10, r=10, t=60, b=10),
#         plot_bgcolor="white",
#         legend=dict(
#             orientation="h",
#             yanchor="top",
#             y=1.12,
#             xanchor="center",
#             x=0.5,
#         ),
#     )
#     fig.update_xaxes(title="Moyenne", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
#     fig.update_yaxes(title="Écart-type", showgrid=True, gridcolor="rgba(0,0,0,0.06)")

#     return fig


import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Colonnes candidates pour l'écart-type (adapte si ton nom exact diffère)
STD_COL_CANDIDATES = [
    "ecart_type", "ecart-type", "std", "sigma",
    "ecart_type_eleves", "ecart_type_resultats"
]

def pick_std_col(df: pd.DataFrame) -> str | None:
    cols = set(df.columns)
    for c in STD_COL_CANDIDATES:
        if c in cols:
            return c
    return None


def agg_mean_std_by_etab_session(
    df_scope: pd.DataFrame,
    exam: str,
    sessions: list[int],
    group_col: str | None = None,      # "bloc_epreuve" ou "epreuve"
    group_value: str | None = None,    # valeur de bloc/épreuve sélectionnée
) -> pd.DataFrame:
    """
    Sortie: 1 ligne par (session, etablissement) avec mean(moyenne) + mean(ecart_type)
    Si group_col/group_value fournis -> filtre sur ce groupe avant agrégation.
    """
    if df_scope.empty:
        return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

    std_col = pick_std_col(df_scope)
    if std_col is None:
        return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

    d = df_scope[(df_scope["examen_u"] == exam) & (df_scope["session"].isin(sessions))].copy()

    if group_col is not None and group_value is not None:
        d = d[d[group_col].astype(str) == str(group_value)].copy()

    d = d.dropna(subset=["moyenne", std_col, "etablissement", "session"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["session", "etablissement", "mean", "std"])

    out = (
        d.groupby(["session", "etablissement"], as_index=False)
         .agg(mean=("moyenne", "mean"), std=(std_col, "mean"))
    )
    out["session"] = out["session"].astype(int)
    return out


def add_iso_ratio_guides(fig: go.Figure, x_min: float, x_max: float, ratios=(6, 7, 8, 9, 10)):
    """Ajoute des lignes iso-ratio μ/σ (cadrans visuels) sur le scatter."""
    xs = np.linspace(x_min, x_max, 80)
    for r in ratios:
        ys = xs / float(r)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1, color="rgba(0,0,0,0.18)", dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=float(xs[-1]),
            y=float(ys[-1]),
            text=f"μ/σ={r}",
            showarrow=False,
            font=dict(size=10, color="rgba(0,0,0,0.45)"),
            xanchor="left",
            yanchor="middle",
        )

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def make_mean_std_scatter(
    df_scope: pd.DataFrame,
    exam: str,
    sessions: list[int],
    etab_selected: str,
    color_map: dict[str, str],
    group_col: str | None = None,
    group_value: str | None = None,
    only_one_year: int | None = None,
    show_network_trend: bool = True,   # trajectoire barycentre réseau (privacy-safe)
    show_change_badge: bool = True,    # badge Δσ 1ère->dernière vs réseau (privacy-safe)
    show_quadrants: bool = True,       # zones pédagogiques
):
    """
    Scatter Moyenne (x) vs Écart-type (y).
    - Réseau: points colorés par session, hover désactivé (confidentialité)
    - Sélection: points + ligne, hover complet + badge TOP%
    - Réseau (moy.): trajectoire du barycentre (optionnel)
    - Quadrants pédagogiques (optionnel): lignes mean/std réseau + zone "🟢 Bas-Droite"
    - Badge Δσ (optionnel): évolution dispersion sélection vs distribution réseau (privacy-safe)
    """
    # --- agrégation attendue en amont: 1 ligne par (session, etablissement) ---
    # df_scope doit contenir: examen_u, session, etablissement, moyenne, + colonne std candidate
    # On suppose que tu as déjà agg_mean_std_by_etab_session dans ton fichier utils.

    sess = sessions if only_one_year is None else [int(only_one_year)]

    df_plot = agg_mean_std_by_etab_session(
        df_scope=df_scope,
        exam=exam,
        sessions=sess,
        group_col=group_col,
        group_value=group_value,
    )
    if df_plot.empty:
        return None

    df_plot["rank"] = df_plot.groupby("session")["mean"].rank(ascending=False, method="min").astype(int)
    df_sel = df_plot[df_plot["etablissement"] == etab_selected].copy()
    df_oth = df_plot[df_plot["etablissement"] != etab_selected].copy()

    fig = go.Figure()

    # ---- Quadrants pédagogiques (basé sur "centre" réseau global) ----
    if show_quadrants:
        mean_net = float(df_plot["mean"].mean())
        std_net = float(df_plot["std"].mean())

        # Zone "🟢 Performance homogène" = Bas-Droite
        fig.add_shape(
            type="rect",
            x0=mean_net, x1=float(df_plot["mean"].max()) + 0.001,
            y0=float(df_plot["std"].min()) - 0.001, y1=std_net,
            fillcolor="rgba(0,200,100,0.06)",
            line=dict(width=0),
            layer="below",
        )

        # Lignes de séparation
        fig.add_vline(x=mean_net, line=dict(color="rgba(0,0,0,0.18)", dash="dash"))
        fig.add_hline(y=std_net, line=dict(color="rgba(0,0,0,0.18)", dash="dash"))

        # Labels simples
        fig.add_annotation(
            x=0.01, y=0.02, xref="paper", yref="paper",
            text="🟢 Moyenne ↑ & Dispersion ↓ (zone idéale)",
            showarrow=False,
            font=dict(size=11, color="rgba(0,0,0,0.65)"),
            align="left",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.06)",
            borderwidth=1,
            borderpad=6,
        )

    # ---- Réseau: points par session (hover OFF) ----
    for s in sorted(df_oth["session"].unique().tolist()):
        serie = f"{exam} {int(s)}"
        dfx = df_oth[df_oth["session"] == s]

        fig.add_trace(
            go.Scatter(
                x=dfx["mean"],
                y=dfx["std"],
                mode="markers",
                name=serie,
                marker=dict(
                    size=10,
                    color=color_map.get(serie, "#B0B0B0"),
                    opacity=0.55,
                    line=dict(width=0),
                ),
                hoverinfo="skip",  # CONFIDENTIALITÉ
                showlegend=True if len(sess) > 1 else False,
            )
        )

    # ---- Trajectoire barycentre réseau (privacy-safe) ----
    if show_network_trend:
        net = (
            df_plot.groupby("session", as_index=False)
                   .agg(mean_net=("mean", "mean"), std_net=("std", "mean"))
                   .sort_values("session")
        )
        fig.add_trace(
            go.Scatter(
                x=net["mean_net"],
                y=net["std_net"],
                mode="lines+markers",
                name="Réseau (moy.)",
                line=dict(width=2, color="rgba(0,0,0,0.35)"),
                marker=dict(size=10, color="rgba(0,0,0,0.35)", line=dict(width=1, color="white")),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # ---- Sélection: points + ligne + hover complet ----
    if not df_sel.empty:
        df_sel = df_sel.sort_values("session")

        n_by_sess = df_plot.groupby("session")["etablissement"].nunique()
        df_sel["top_pct"] = df_sel.apply(
            lambda r: (int(r["rank"]) / int(n_by_sess.get(r["session"], np.nan))) * 100, axis=1
        )

        fig.add_trace(
            go.Scatter(
                x=df_sel["mean"],
                y=df_sel["std"],
                mode="lines+markers",
                name="Sélection",
                line=dict(color="#111827", width=2.5),
                marker=dict(
                    size=16,
                    color="#111827",
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                customdata=np.stack([df_sel["rank"], df_sel["session"], df_sel["top_pct"]], axis=-1),
                hovertemplate=(
                    f"<b>{etab_selected}</b><br>"
                    "Session: %{customdata[1]}<br>"
                    "Moyenne: %{x:.2f}<br>"
                    "Écart-type: %{y:.2f}<br>"
                    "Rang: %{customdata[0]}<br>"
                    "Top: %{customdata[2]:.0f}%<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Badge TOP% sur le dernier point
        last = df_sel.iloc[-1]
        fig.add_annotation(
            x=float(last["mean"]),
            y=float(last["std"]),
            text=f"<b>{etab_selected}</b><br>TOP {float(last['top_pct']):.0f}%",
            showarrow=True,
            arrowhead=0,
            ax=0, ay=-45,
            font=dict(color="white", size=11),
            bgcolor="#FF4B4B",
            borderpad=6,
        )

        # ---- Badge "Δσ vs réseau" : réponse directe à “ma dispersion évolue plus/moins que les autres” ----
        if show_change_badge and len(df_sel) >= 2:
            year_a = int(df_sel["session"].iloc[0])
            year_b = int(df_sel["session"].iloc[-1])

            piv = df_plot[df_plot["session"].isin([year_a, year_b])].pivot_table(
                index="etablissement", columns="session", values=["mean", "std"], aggfunc="first"
            )

            if (year_a in piv["std"].columns) and (year_b in piv["std"].columns) and (etab_selected in piv.index):
                delta_std = (piv["std"][year_b] - piv["std"][year_a]).dropna()
                delta_mean = (piv["mean"][year_b] - piv["mean"][year_a]).dropna()

                common = delta_std.index.intersection(delta_mean.index)
                delta_std = delta_std.loc[common]
                delta_mean = delta_mean.loc[common]

                if not delta_std.empty:
                    sel_dstd = float(delta_std.loc[etab_selected])
                    sel_dmean = float(delta_mean.loc[etab_selected])
                    n = int(delta_std.shape[0])

                    # percentile: % d'étabs dont Δσ est inférieur au tien
                    pct_higher_std = 100.0 * float((delta_std < sel_dstd).sum()) / n

                    # Message simple
                    if sel_dstd >= 0:
                        msg_std = f"Δσ {year_a}→{year_b}: +{sel_dstd:.2f}<br>+ que {pct_higher_std:.0f}% du réseau"
                    else:
                        msg_std = f"Δσ {year_a}→{year_b}: {sel_dstd:.2f}<br>↓ vs {pct_higher_std:.0f}% du réseau"

                    # Bonus lecture moyenne
                    if sel_dmean >= 0:
                        msg_mu = f"Δμ {year_a}→{year_b}: +{sel_dmean:.2f}"
                    else:
                        msg_mu = f"Δμ {year_a}→{year_b}: {sel_dmean:.2f}"

                    fig.add_annotation(
                        x=0.01, y=0.98, xref="paper", yref="paper",
                        text=f"<b>Lecture rapide</b><br>{msg_mu}<br>{msg_std}",
                        showarrow=False,
                        align="left",
                        font=dict(size=11, color="rgba(0,0,0,0.85)"),
                        bgcolor="rgba(255,255,255,0.88)",
                        bordercolor="rgba(0,0,0,0.08)",
                        borderwidth=1,
                        borderpad=6,
                    )

    title = "Moyenne vs Écart-type"
    if group_col and group_value:
        title += f" — {group_value}"

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(title="Moyenne", showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(title="Écart-type", showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    return fig
