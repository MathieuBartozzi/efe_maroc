# # =========================================================
# # PAGES/XX_ETABLISSEMENT.PY  (Streamlit multipage)
# # =========================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px

# G10 = px.colors.qualitative.G10

# # -------------------------
# # Paramètres
# # -------------------------
# WANTED_SESSIONS = [2023, 2024, 2025]     # tu veux explicitement 23/24/25
# TOP_N = 6                               # nombre de métriques "détails" affichées par examen

# BAC_EXCLUDE = {"GO", "GRAND ORAL", "PHILO", "PHILOSOPHIE", "FRANÇAIS", "FRANCAIS"}
# DNB_EPREUVE_EXCLUDE = {                 # optionnel: reprends tes exclusions DNB si tu veux
#     "GRAMMAIRE ET COMPRÉHENSION",
#     "DICTÉE",
#     "RÉDACTION",
# }

# # Couleurs (G10) : vert / orange-rouge
# SUCCESS_COLOR = G10[2]  # généralement un vert
# FAIL_COLOR = G10[1]     # généralement orange/rouge

# # -------------------------
# # Helpers
# # -------------------------
# def norm_series(s: pd.Series) -> pd.Series:
#     return (
#         s.astype(str)
#          .str.upper()
#          .str.strip()
#          .str.replace(r"\s+", " ", regex=True)
#     )

# def safe_mean_by_year(dfi: pd.DataFrame, value_col="moyenne") -> pd.Series:
#     if dfi.empty:
#         return pd.Series(dtype="float64")
#     return dfi.groupby("session")[value_col].mean().sort_index()

# def metric_current_delta(series_by_year: pd.Series, year_cur: int, year_prev: int):
#     v_cur = series_by_year.get(year_cur, np.nan)
#     v_prev = series_by_year.get(year_prev, np.nan)
#     delta = np.nan if (pd.isna(v_cur) or pd.isna(v_prev)) else float(v_cur - v_prev)
#     return v_cur, delta

# def agg_group_session_mean(dfi: pd.DataFrame, group_col: str) -> pd.DataFrame:
#     if dfi.empty:
#         return pd.DataFrame(columns=[group_col, "session", "mean"])
#     return (
#         dfi.groupby([group_col, "session"], dropna=False)["moyenne"]
#            .mean()
#            .reset_index(name="mean")
#     )

# def build_delta_df(
#     df_all: pd.DataFrame,
#     df_etab: pd.DataFrame,
#     exam: str,
#     group_col: str,
#     sessions: list[int],
#     exclude_values: set[str] | None = None,
#     exclude_col: str | None = None,
# ) -> pd.DataFrame:
#     """
#     Retourne un DF avec:
#       exam, group_label, session, delta = mean(etab) - mean(reseau)
#     où group_label = valeur de group_col (bloc_epreuve pour BAC, epreuve pour DNB)
#     """

#     d_all = df_all[df_all["examen_u"] == exam].copy()
#     d_etb = df_etab[df_etab["examen_u"] == exam].copy()

#     # Filtre sessions
#     d_all = d_all[d_all["session"].isin(sessions)].copy()
#     d_etb = d_etb[d_etb["session"].isin(sessions)].copy()

#     # Exclusions
#     if exclude_values and exclude_col:
#         d_all["ex_u"] = norm_series(d_all[exclude_col])
#         d_etb["ex_u"] = norm_series(d_etb[exclude_col])
#         d_all = d_all[~d_all["ex_u"].isin(exclude_values)].copy()
#         d_etb = d_etb[~d_etb["ex_u"].isin(exclude_values)].copy()

#     d_all = d_all[d_all[group_col].notna()].copy()
#     d_etb = d_etb[d_etb[group_col].notna()].copy()

#     # Agrégations
#     net = agg_group_session_mean(d_all, group_col=group_col)
#     etb = agg_group_session_mean(d_etb, group_col=group_col)

#     if net.empty and etb.empty:
#         return pd.DataFrame(columns=["exam", "group_label", "session", "delta"])

#     # Merge sur group/session
#     merged = pd.merge(
#         etb.rename(columns={"mean": "mean_etab"}),
#         net.rename(columns={"mean": "mean_net"}),
#         on=[group_col, "session"],
#         how="outer",
#     )

#     merged["delta"] = merged["mean_etab"] - merged["mean_net"]
#     merged["exam"] = exam
#     merged = merged.rename(columns={group_col: "group_label"})

#     # Ordre: BAC puis DNB (géré au moment de construire x)
#     return merged[["exam", "group_label", "session", "delta"]].dropna(subset=["delta"])

# def make_delta_bar_figure(delta_df: pd.DataFrame, sessions: list[int]) -> go.Figure:
#     """
#     Barres groupées par session sur x = (BAC: bloc) + (DNB: epreuve).
#     Couleur par signe (delta).
#     """

#     if delta_df.empty:
#         fig = go.Figure()
#         fig.update_layout(
#             height=420,
#             margin=dict(l=0, r=0, t=20, b=0),
#             xaxis_title="",
#             yaxis_title="Écart (Établissement – Réseau)",
#         )
#         fig.add_annotation(
#             text="Aucune donnée disponible pour cet établissement sur la période.",
#             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
#         )
#         return fig

#     # Construire un x_label unique, avec BAC d’abord, DNB ensuite
#     # (et stable)
#     bac_labels = sorted(delta_df.loc[delta_df["exam"] == "BAC", "group_label"].dropna().unique().tolist())
#     dnb_labels = sorted(delta_df.loc[delta_df["exam"] == "DNB", "group_label"].dropna().unique().tolist())

#     x_order = [f"BAC — {x}" for x in bac_labels] + [f"DNB — {x}" for x in dnb_labels]

#     # Ajout x_label dans DF
#     delta_df = delta_df.copy()
#     delta_df["x_label"] = np.where(
#         delta_df["exam"] == "BAC",
#         "BAC — " + delta_df["group_label"].astype(str),
#         "DNB — " + delta_df["group_label"].astype(str),
#     )

#     fig = go.Figure()

#     # Barres par session
#     for i, s in enumerate(sessions):
#         d = delta_df[delta_df["session"] == s].copy()
#         if d.empty:
#             continue

#         d["x_label"] = pd.Categorical(d["x_label"], categories=x_order, ordered=True)
#         d = d.sort_values("x_label")

#         colors = [SUCCESS_COLOR if v >= 0 else FAIL_COLOR for v in d["delta"].fillna(0).tolist()]

#         fig.add_trace(
#             go.Bar(
#                 name=str(s),
#                 x=d["x_label"].astype(str),
#                 y=d["delta"],
#                 marker_color=colors,
#                 opacity=0.95,
#             )
#         )

#     # Ligne zéro
#     fig.add_hline(y=0, line_width=1)

#     fig.update_layout(
#         barmode="group",
#         bargap=0.15,
#         bargroupgap=0.05,
#         height=520,
#         margin=dict(l=0, r=0, t=20, b=0),
#         legend_title_text="",
#         xaxis_title="",
#         yaxis_title="Écart (Établissement – Réseau)",
#     )
#     fig.update_xaxes(tickangle=-25)

#     return fig

# def render_metric_grid(title: str, items: list[tuple[str, float, float]], cols: int = 3):
#     """
#     items: list of (label, value, delta)
#     """
#     st.subheader(title)
#     if not items:
#         st.caption("Aucune donnée.")
#         return
#     rows = (len(items) + cols - 1) // cols
#     idx = 0
#     for _ in range(rows):
#         cs = st.columns(cols)
#         for c in cs:
#             if idx >= len(items):
#                 break
#             label, value, delta = items[idx]
#             c.metric(
#                 label=label,
#                 value="—" if pd.isna(value) else f"{value:.2f}",
#                 delta=None if pd.isna(delta) else f"{delta:+.2f}",
#                 border=True,
#             )
#             idx += 1

# # =========================================================
# # MAIN
# # =========================================================
# st.title("Page Établissement")

# if "df" not in st.session_state:
#     st.error("Aucune donnée chargée dans st.session_state['df'].")
#     st.stop()

# df = st.session_state["df"].copy()
# df["session"] = df["session"].astype(int)

# # Normalisation minimale
# df["examen_u"] = norm_series(df["examen"])

# # Sélection établissement
# st.sidebar.header("Sélection")
# etabs = sorted(df["etablissement"].dropna().unique().tolist())
# etab = st.sidebar.selectbox("Établissement", etabs)

# # Sessions: forcer 23/24/25 si dispo
# sessions_all = sorted(df["session"].dropna().unique().tolist())
# SESSIONS = [s for s in WANTED_SESSIONS if s in sessions_all]
# if len(SESSIONS) < 2:
#     SESSIONS = sessions_all[-3:] if len(sessions_all) >= 3 else sessions_all

# year_cur = max(SESSIONS) if SESSIONS else None
# year_prev = sorted(SESSIONS)[-2] if len(SESSIONS) >= 2 else None

# st.sidebar.caption(f"Sessions : {', '.join(map(str, SESSIONS))}")

# # DFs réseau + établissement
# df_all = df[df["session"].isin(SESSIONS)].copy()
# df_etab = df_all[df_all["etablissement"] == etab].copy()

# # Détection présence examens pour l'établissement (responsive)
# has_bac = (df_etab["examen_u"] == "BAC").any()
# has_dnb = (df_etab["examen_u"] == "DNB").any()

# st.caption(f"Établissement sélectionné : **{etab}**")

# # =========================================================
# # 1) KPIs haut de page (BAC / DNB_FINAL)
# # =========================================================
# st.subheader("Indicateurs clés")

# kpi_cols = st.columns(3)

# with kpi_cols[0]:
#     # tu veux border=True partout
#     kpi_cols[0].metric("Sessions", ", ".join(map(str, SESSIONS)), border=True)

# with kpi_cols[1]:
#     if has_bac and year_cur is not None and year_prev is not None:
#         bac_series = safe_mean_by_year(df_etab[df_etab["examen_u"] == "BAC"])
#         v, d = metric_current_delta(bac_series, year_cur, year_prev)
#         st.metric("Moyenne BAC", "—" if pd.isna(v) else f"{v:.2f}", None if pd.isna(d) else f"{d:+.2f}", border=True)
#     else:
#         st.metric("Moyenne BAC", "—", None, border=True)

# with kpi_cols[2]:
#     if has_dnb and year_cur is not None and year_prev is not None:
#         # DNB global = DNB_FINAL uniquement (comme ton script)
#         dnb_mask = (df_etab["examen_u"] == "DNB") & (norm_series(df_etab["bloc_epreuve"]) == "DNB_FINAL")
#         dnb_series = safe_mean_by_year(df_etab[dnb_mask])
#         v, d = metric_current_delta(dnb_series, year_cur, year_prev)
#         st.metric("Moyenne DNB (FINAL)", "—" if pd.isna(v) else f"{v:.2f}", None if pd.isna(d) else f"{d:+.2f}", border=True)
#     else:
#         st.metric("Moyenne DNB (FINAL)", "—", None, border=True)

# # =========================================================
# # 2) Métriques "détails" (sans empilement) -> expanders
# #    - BAC: par bloc_epreuve
# #    - DNB: par epreuve
# # =========================================================
# st.divider()

# with st.expander("Détails BAC — blocs d’épreuves (Top variations)", expanded=False):
#     if not has_bac or year_cur is None or year_prev is None:
#         st.caption("Aucune donnée BAC pour cet établissement.")
#     else:
#         d_bac = df_etab[df_etab["examen_u"] == "BAC"].copy()
#         d_bac = d_bac[d_bac["bloc_epreuve"].notna()].copy()
#         d_bac["bloc_u"] = norm_series(d_bac["bloc_epreuve"])
#         d_bac = d_bac[~d_bac["bloc_u"].isin(BAC_EXCLUDE)].copy()

#         # mean par bloc x année
#         p = (
#             d_bac.groupby(["bloc_epreuve", "session"])["moyenne"]
#                  .mean()
#                  .reset_index(name="mean")
#         )
#         wide = p.pivot(index="bloc_epreuve", columns="session", values="mean")
#         v_cur = wide.get(year_cur)
#         v_prev = wide.get(year_prev)
#         delta = v_cur - v_prev

#         tmp = pd.DataFrame({
#             "label": wide.index,
#             "value": v_cur.values if v_cur is not None else np.nan,
#             "delta": delta.values if delta is not None else np.nan,
#         }).dropna(subset=["value"])

#         # Top N hausses + Top N baisses
#         tmp_sorted = tmp.sort_values("delta", ascending=False)
#         top_plus = tmp_sorted.head(TOP_N)
#         top_minus = tmp_sorted.tail(TOP_N).sort_values("delta", ascending=True)

#         st.caption(f"Affiché : Top {TOP_N} hausses et Top {TOP_N} baisses ({year_prev} → {year_cur})")

#         items = []
#         for _, r in pd.concat([top_plus, top_minus], ignore_index=True).iterrows():
#             items.append((str(r["label"]), float(r["value"]), float(r["delta"]) if pd.notna(r["delta"]) else np.nan))

#         render_metric_grid("BAC — Blocs", items, cols=3)

# with st.expander("Détails DNB — épreuves (Top variations)", expanded=False):
#     if not has_dnb or year_cur is None or year_prev is None:
#         st.caption("Aucune donnée DNB pour cet établissement.")
#     else:
#         d_dnb = df_etab[df_etab["examen_u"] == "DNB"].copy()
#         d_dnb = d_dnb[d_dnb["epreuve"].notna()].copy()
#         d_dnb["ep_u"] = norm_series(d_dnb["epreuve"])
#         d_dnb = d_dnb[~d_dnb["ep_u"].isin(DNB_EPREUVE_EXCLUDE)].copy()

#         p = (
#             d_dnb.groupby(["epreuve", "session"])["moyenne"]
#                  .mean()
#                  .reset_index(name="mean")
#         )
#         wide = p.pivot(index="epreuve", columns="session", values="mean")
#         v_cur = wide.get(year_cur)
#         v_prev = wide.get(year_prev)
#         delta = v_cur - v_prev

#         tmp = pd.DataFrame({
#             "label": wide.index,
#             "value": v_cur.values if v_cur is not None else np.nan,
#             "delta": delta.values if delta is not None else np.nan,
#         }).dropna(subset=["value"])

#         tmp_sorted = tmp.sort_values("delta", ascending=False)
#         top_plus = tmp_sorted.head(TOP_N)
#         top_minus = tmp_sorted.tail(TOP_N).sort_values("delta", ascending=True)

#         st.caption(f"Affiché : Top {TOP_N} hausses et Top {TOP_N} baisses ({year_prev} → {year_cur})")

#         items = []
#         for _, r in pd.concat([top_plus, top_minus], ignore_index=True).iterrows():
#             items.append((str(r["label"]), float(r["value"]), float(r["delta"]) if pd.notna(r["delta"]) else np.nan))

#         render_metric_grid("DNB — Épreuves", items, cols=3)

# # =========================================================
# # 3) Graphique principal : écarts établissement – réseau
# #    BAC: bloc_epreuve ; DNB: epreuve
# # =========================================================
# st.divider()
# st.subheader("Écarts vs Réseau (Établissement – Réseau)")

# delta_parts = []

# if has_bac:
#     delta_bac = build_delta_df(
#         df_all=df_all,
#         df_etab=df_etab,
#         exam="BAC",
#         group_col="bloc_epreuve",
#         sessions=SESSIONS,
#         exclude_values=BAC_EXCLUDE,
#         exclude_col="bloc_epreuve",
#     )
#     delta_parts.append(delta_bac)

# if has_dnb:
#     delta_dnb = build_delta_df(
#         df_all=df_all,
#         df_etab=df_etab,
#         exam="DNB",
#         group_col="epreuve",
#         sessions=SESSIONS,
#         exclude_values=DNB_EPREUVE_EXCLUDE,
#         exclude_col="epreuve",
#     )
#     delta_parts.append(delta_dnb)

# delta_df = pd.concat(delta_parts, ignore_index=True) if delta_parts else pd.DataFrame()

# fig = make_delta_bar_figure(delta_df, sessions=SESSIONS)
# st.plotly_chart(fig, use_container_width=True)












# import streamlit as st
# import pandas as pd
# import numpy as np

# DNB_EPREUVE_EXCLUDE = {
#     "DICTÉE",
#     "DICTEE",
#     "RÉDACTION",
#     "REDACTION",
#     "GRAMMAIRE ET COMPRÉHENSION",
#     "GRAMMAIRE ET COMPREHENSION",
# }

# def build_rank_pivot(df_scope: pd.DataFrame, group_col: str) -> pd.DataFrame:
#     """
#     Rang = position de l'établissement (1 = meilleur) parmi tous les établissements,
#     pour chaque (group_col, session).
#     """
#     if df_scope.empty:
#         return pd.DataFrame()

#     tmp = (
#         df_scope.groupby([group_col, "session", "etablissement"], dropna=False)["moyenne"]
#                .mean()
#                .reset_index(name="mean")
#     )

#     tmp["rank"] = tmp.groupby([group_col, "session"])["mean"].rank(
#         ascending=False, method="min"
#     )

#     return tmp.pivot_table(
#         index=[group_col, "etablissement"],
#         columns="session",
#         values="rank",
#         aggfunc="first",
#     )

# def get_rank_row(rank_pivot: pd.DataFrame, group_col: str, group_value: str, etab: str) -> pd.DataFrame:
#     """
#     Renvoie un DF 1 ligne avec colonnes 23/24/25 (rang).
#     """
#     def fmt(x):
#         return "—" if pd.isna(x) else str(int(round(float(x))))

#     if rank_pivot is None or rank_pivot.empty:
#         return pd.DataFrame([{"23": "—", "24": "—", "25": "—"}])

#     key = (group_value, etab)
#     if key not in rank_pivot.index:
#         return pd.DataFrame([{"23": "—", "24": "—", "25": "—"}])

#     s = rank_pivot.loc[key]
#     return pd.DataFrame([{
#         "23": fmt(s.get(2023, np.nan)),
#         "24": fmt(s.get(2024, np.nan)),
#         "25": fmt(s.get(2025, np.nan)),
#     }])
# # ----------------------------
# # Guard
# # ----------------------------
# if "df" not in st.session_state:
#     st.error("Aucune donnée chargée dans st.session_state['df'].")
#     st.stop()

# df = st.session_state["df"].copy()
# df["session"] = df["session"].astype(int)

# # ----------------------------
# # Paramètres fixes
# # ----------------------------
# SESSIONS = [2023, 2024, 2025]
# YEAR_CUR = 2025
# YEAR_PREV = 2024

# # ----------------------------
# # Helpers
# # ----------------------------
# def norm_series(s: pd.Series) -> pd.Series:
#     return (
#         s.astype(str)
#          .str.upper()
#          .str.strip()
#          .str.replace(r"\s+", " ", regex=True)
#     )

# def series_23_25(dfi: pd.DataFrame, value_col: str = "moyenne") -> pd.Series:
#     """Retourne une série indexée par session (2023,2024,2025) avec mean, NaN si absent."""
#     if dfi.empty:
#         return pd.Series(index=SESSIONS, dtype="float64")
#     s = dfi.groupby("session")[value_col].mean()
#     return pd.Series({y: float(s.get(y, np.nan)) for y in SESSIONS})

# def current_and_delta(s: pd.Series) -> tuple[float, float]:
#     """(valeur_2025, delta_2025_vs_2024) avec NaN si non dispo."""
#     v_cur = s.get(YEAR_CUR, np.nan)
#     v_prev = s.get(YEAR_PREV, np.nan)
#     delta = np.nan if (pd.isna(v_cur) or pd.isna(v_prev)) else float(v_cur - v_prev)
#     return v_cur, delta

# import altair as alt
# import pandas as pd
# import numpy as np
# import streamlit as st

# SESSIONS = [2023, 2024, 2025]
# YEAR_CUR = 2025
# YEAR_PREV = 2024

# import altair as alt
# import pandas as pd
# import numpy as np
# import streamlit as st

# SESSIONS = [2023, 2024, 2025]
# YEAR_CUR = 2025
# YEAR_PREV = 2024


# def render_card(title: str, s: pd.Series, rank_df: pd.DataFrame, y_domain=(9, 18)):
#     if s.isna().all():
#         return

#     v23 = s.get(2023, np.nan)
#     v24 = s.get(2024, np.nan)
#     v25 = s.get(2025, np.nan)

#     value_str = "Donnée 2025 indisponible" if pd.isna(v25) else f"{float(v25):.2f}"

#     if pd.isna(v25) or pd.isna(v24):
#         delta_str = "+0.00"
#     else:
#         delta_str = f"{float(v25 - v24):+.2f}"

#     with st.container(border=True):
#         st.metric(title, value=value_str, delta=delta_str, border=True)

#         # mini chart (Altair) — conserve 23-24 si 25 manque
#         chart_df = pd.DataFrame(
#             {"session": ["2023", "2024", "2025"], "mean": [v23, v24, v25]}
#         ).dropna(subset=["mean"])

#         chart = (
#             alt.Chart(chart_df)
#             .mark_line(point=True)
#             .encode(
#                 x=alt.X("session:N", title=None, axis=alt.Axis(labelAngle=-45)),
#                 y=alt.Y("mean:Q", title=None, scale=alt.Scale(domain=list(y_domain))),
#             )
#             .properties(height=120)
#         )
#         st.altair_chart(chart, use_container_width=True)

#         # mini tableau rangs (1 ligne)
#         st.caption("Rang (réseau)")
#         st.dataframe(rank_df, hide_index=True, use_container_width=True, height=70)



# # ----------------------------
# # Sidebar: sélection établissement
# # ----------------------------
# st.sidebar.header("Sélection")
# etabs = sorted(df["etablissement"].dropna().unique().tolist())
# etab = st.sidebar.selectbox("Établissement", etabs)

# # Filtre établissement + sessions
# df_e = df[(df["etablissement"] == etab) & (df["session"].isin(SESSIONS))].copy()
# df_e["examen_u"] = norm_series(df_e["examen"])
# df_e["bloc_u"] = norm_series(df_e["bloc_epreuve"])
# df_e["epreuve_u"] = norm_series(df_e["epreuve"])

# # ----------------------------
# # KPIs haut de page (seulement BAC + DNB_FINAL)
# # ----------------------------
# st.title(f"Établissement : {etab}")
# st.subheader("Indicateurs clés")

# c1, c2 = st.columns(2)

# with c1:
#     # BAC global
#     s_bac = series_23_25(df_e[df_e["examen_u"] == "BAC"])
#     v, d = current_and_delta(s_bac)
#     st.metric(
#         "Moyenne BAC",
#         value="—" if pd.isna(v) else f"{v:.2f}",
#         delta=None if pd.isna(d) else f"{d:+.2f}",
#         border=True,
#     )

# with c2:
#     # DNB global = DNB_FINAL uniquement
#     dnb_mask = (df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")
#     s_dnb = series_23_25(df_e[dnb_mask])
#     v, d = current_and_delta(s_dnb)
#     st.metric(
#         "Moyenne DNB (FINAL)",
#         value="—" if pd.isna(v) else f"{v:.2f}",
#         delta=None if pd.isna(d) else f"{d:+.2f}",
#         border=True,
#     )

# st.divider()

# # ----------------------------
# # Tabs: BAC finals / DNB finals / Spécialités
# # ----------------------------
# tab_bac, tab_dnb, tab_spe = st.tabs(
#     ["BAC — Épreuves finales", "DNB — Épreuves finales", "BAC — Spécialités"]
# )

# # =========================================================
# # TAB 1 — BAC Épreuves finales (cartes par bloc_epreuve)
# # =========================================================
# with tab_bac:
#     bac = df_e[df_e["examen_u"] == "BAC"].copy()

#     # On cible explicitement les blocs demandés
#     # Remarque: adapte si tes libellés diffèrent (ex: "GRAND ORAL" vs "GO")
#     target_blocs = [
#         ("EDS", {"EDS"}),  # bloc EDS
#         ("EAF", {"EAF"}),  # épreuves anticipées (Français) si tu l'as sous ce code
#         ("Grand Oral", {"GO", "GRAND ORAL"}),
#         ("Philosophie", {"PHILO", "PHILOSOPHIE"}),
#     ]

#     card_items = []
#     for label, accepted in target_blocs:
#         m = bac["bloc_u"].isin(accepted)
#         s = series_23_25(bac[m])
#         card_items.append((label, s))

#     render_cards_grid(card_items, cols=4)

# # =========================================================
# # TAB 2 — DNB Épreuves finales (cartes par matière = epreuve)
# # =========================================================
# with tab_dnb:
#     dnb = df_e[
#     (df_e["examen_u"] == "DNB") &
#     (df_e["bloc_u"] == "DNB_FINAL")
#     ].copy()

#     # Normalisation + exclusion Dictée / Rédaction / Grammaire
#     dnb["ep_u"] = norm_series(dnb["epreuve"])
#     dnb = dnb[~dnb["ep_u"].isin(DNB_EPREUVE_EXCLUDE)].copy()

#     if dnb.empty:
#         # Pas de message, comme demandé
#         pass
#     else:
#         # Une carte par matière (epreuve)
#         card_items = []
#         for epr in sorted(dnb["epreuve"].dropna().unique().tolist()):
#             s = series_23_25(dnb[dnb["epreuve"] == epr])
#             card_items.append((str(epr), s))

#         render_cards_grid(card_items, cols=5)

# # =========================================================
# # TAB 3 — BAC Spécialités (cartes par spécialité)
# # -> Interprétation la plus cohérente avec ton modèle :
# #    spécialités = epreuves (matières) du bloc EDS
# # =========================================================
# with tab_spe:
#     bac = df_e[df_e["examen_u"] == "BAC"].copy()
#     spe = bac[bac["bloc_u"] == "EDS"].copy()

#     if spe.empty:
#         # Pas de message, comme demandé
#         pass
#     else:
#         card_items = []
#         for epr in sorted(spe["epreuve"].dropna().unique().tolist()):
#             s = series_23_25(spe[spe["epreuve"] == epr])
#             card_items.append((str(epr), s))

#         render_cards_grid(card_items, cols=4)


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =========================================================
# PARAMÈTRES / EXCLUSIONS
# =========================================================
SESSIONS = [2023, 2024, 2025]
YEAR_CUR = 2025
YEAR_PREV = 2024

DNB_EPREUVE_EXCLUDE = {
    "DICTÉE", "DICTEE",
    "RÉDACTION", "REDACTION",
    "GRAMMAIRE ET COMPRÉHENSION", "GRAMMAIRE ET COMPREHENSION",
}

# =========================================================
# HELPERS
# =========================================================
def norm_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.upper()
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )

def series_23_25(dfi: pd.DataFrame, value_col: str = "moyenne") -> pd.Series:
    """Série mean par session sur {2023,2024,2025}. NaN si absent."""
    if dfi.empty:
        return pd.Series(index=SESSIONS, dtype="float64")
    s = dfi.groupby("session")[value_col].mean()
    return pd.Series({y: float(s.get(y, np.nan)) for y in SESSIONS})

def current_and_delta(s: pd.Series) -> tuple[float, float]:
    """(valeur_2025, delta_2025_vs_2024)"""
    v_cur = s.get(YEAR_CUR, np.nan)
    v_prev = s.get(YEAR_PREV, np.nan)
    delta = np.nan if (pd.isna(v_cur) or pd.isna(v_prev)) else float(v_cur - v_prev)
    return v_cur, delta

# --------- RANGS (réseau) ---------
def build_rank_pivot(df_scope: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Rang = position de l'établissement (1 = meilleur) parmi tous les établissements,
    pour chaque (group_col, session).
    """
    if df_scope.empty:
        return pd.DataFrame()

    tmp = (
        df_scope.groupby([group_col, "session", "etablissement"], dropna=False)["moyenne"]
               .mean()
               .reset_index(name="mean")
    )

    tmp["rank"] = tmp.groupby([group_col, "session"])["mean"].rank(
        ascending=False, method="min"
    )

    return tmp.pivot_table(
        index=[group_col, "etablissement"],
        columns="session",
        values="rank",
        aggfunc="first",
    )

def get_rank_row(rank_pivot: pd.DataFrame, group_col: str, group_value: str, etab: str) -> pd.DataFrame:
    """DF 1 ligne: colonnes 23/24/25 = rang. '—' si absent."""
    def fmt(x):
        return "—" if pd.isna(x) else str(int(round(float(x))))

    if rank_pivot is None or rank_pivot.empty:
        return pd.DataFrame([{"23": "—", "24": "—", "25": "—"}])

    key = (group_value, etab)
    if key not in rank_pivot.index:
        return pd.DataFrame([{"23": "—", "24": "—", "25": "—"}])

    s = rank_pivot.loc[key]
    return pd.DataFrame([{
        "2023": fmt(s.get(2023, np.nan)),
        "2024": fmt(s.get(2024, np.nan)),
        "2025": fmt(s.get(2025, np.nan)),
    }])

# --------- CARTE + GRILLE ---------
def render_card(title: str, s: pd.Series, rank_df: pd.DataFrame, y_domain=(9, 18)):
    """
    - Si 2025 manquant: value = 'Donnée 2025 indisponible'
    - Delta: +0.00 si non calculable (hauteur stable)
    - Mini chart: conserve 23-24 si 25 manque
    - Ajoute mini tableau des rangs (23/24/25)
    """
    if s.isna().all():
        return

    v23 = s.get(2023, np.nan)
    v24 = s.get(2024, np.nan)
    v25 = s.get(2025, np.nan)

    value_str = "---" if pd.isna(v25) else f"{float(v25):.2f}"
    delta_str = "+0.00" if (pd.isna(v25) or pd.isna(v24)) else f"{float(v25 - v24):+.2f}"

    with st.container(border=True):
        st.metric(title, value=value_str, delta=delta_str, border=True)

        # mini chart (Altair) — conserve 23-24 si 25 manque
        chart_df = pd.DataFrame(
            {"session": ["2023", "2024", "2025"], "mean": [v23, v24, v25]}
        ).dropna(subset=["mean"])

        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("session:N", title=None, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("mean:Q", title=None, scale=alt.Scale(domain=list(y_domain))),
            )
            .properties(height=120)
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption("Rang (réseau)")
        st.dataframe(rank_df, hide_index=True, use_container_width=True, height=70)

def render_cards_grid(card_items: list[tuple[str, pd.Series, pd.DataFrame]], cols: int = 3):
    """Affiche une grille de cartes. Ignore celles dont la série est 100% NaN."""
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
                render_card(title, s, rank_df, y_domain=(9, 18))

# 1) Helper : construire un pivot de rangs + total (N établissements) par session
def build_rank_pivot_with_total(df_scope: pd.DataFrame, group_col: str | None = None):
    """
    Si group_col is None:
        ranking global par établissement et session (mean sur toutes lignes du scope)
        -> index = etablissement, columns=session, values=rank
    Sinon:
        ranking par (group_col, établissement, session)
        -> index=(group_col, etablissement), columns=session, values=rank
    Renvoie (rank_pivot, n_pivot) où n_pivot donne N établissements par session.
    """
    if df_scope.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")

    if group_col is None:
        tmp = (
            df_scope.groupby(["session", "etablissement"], dropna=False)["moyenne"]
                   .mean()
                   .reset_index(name="mean")
        )
        tmp["rank"] = tmp.groupby(["session"])["mean"].rank(ascending=False, method="min")

        rank_pivot = tmp.pivot_table(index="etablissement", columns="session", values="rank", aggfunc="first")
        n_by_session = tmp.groupby("session")["etablissement"].nunique()

        return rank_pivot, n_by_session

    # cas group_col
    tmp = (
        df_scope.groupby([group_col, "session", "etablissement"], dropna=False)["moyenne"]
               .mean()
               .reset_index(name="mean")
    )
    tmp["rank"] = tmp.groupby([group_col, "session"])["mean"].rank(ascending=False, method="min")

    rank_pivot = tmp.pivot_table(index=[group_col, "etablissement"], columns="session", values="rank", aggfunc="first")
    n_by_session = tmp.groupby(["session"])["etablissement"].nunique()

    return rank_pivot, n_by_session


# 2) Helper : formater "rang/total" par année
def format_rank_over_total(rank_val, total_val):
    if pd.isna(rank_val) or pd.isna(total_val) or total_val == 0:
        return "—"
    return f"{int(round(float(rank_val))):d}/{int(total_val):d}"


# 3) Helper : récupérer une ligne de rangs/total pour un établissement
def get_global_rank_row(rank_pivot: pd.DataFrame, n_by_session: pd.Series, etab: str) -> pd.DataFrame:
    """
    Retourne une DF 1 ligne: colonnes 23/24/25 en format 'rang/total'.
    """
    if rank_pivot is None or rank_pivot.empty or etab not in rank_pivot.index:
        return pd.DataFrame([{"23": "—", "24": "—", "25": "—"}])

    s = rank_pivot.loc[etab]
    return pd.DataFrame([{
        "23": format_rank_over_total(s.get(2023, np.nan), n_by_session.get(2023, np.nan)),
        "24": format_rank_over_total(s.get(2024, np.nan), n_by_session.get(2024, np.nan)),
        "25": format_rank_over_total(s.get(2025, np.nan), n_by_session.get(2025, np.nan)),
    }])


#
# =========================================================
# GUARD + CHARGEMENT DATA
# =========================================================
if "df" not in st.session_state:
    st.error("Aucune donnée chargée dans st.session_state['df'].")
    st.stop()

df = st.session_state["df"].copy()
df["session"] = df["session"].astype(int)

# =========================================================
# SIDEBAR: ÉTABLISSEMENT
# =========================================================
st.sidebar.header("Sélection")
etabs = sorted(df["etablissement"].dropna().unique().tolist())
etab = st.sidebar.selectbox("Établissement", etabs)

# Réseau (tous établissements) sur les sessions
df_all = df[df["session"].isin(SESSIONS)].copy()
df_all["examen_u"] = norm_series(df_all["examen"])
df_all["bloc_u"] = norm_series(df_all["bloc_epreuve"])
df_all["epreuve_u"] = norm_series(df_all["epreuve"])

# Établissement sélectionné
df_e = df_all[df_all["etablissement"] == etab].copy()

# =========================================================
# KPIs haut de page
# =========================================================
st.title(f"Établissement : {etab}")
st.subheader("Indicateurs clés")

#=========================================================
# INTÉGRATION DANS TA PAGE : JUSTE APRÈS df_all / df_e
# =========================================================

# --- BAC global (réseau) : rank par établissement sur l'ensemble des lignes BAC ---
bac_net_global = df_all[["examen_u"] == "BAC"].copy()
rank_bac_global, n_bac = build_rank_pivot_with_total(bac_net_global, group_col=None)

# --- DNB global (réseau) : rank par établissement sur DNB_FINAL uniquement ---
dnb_net_global = df_all[(df_all["examen_u"] == "DNB") & (df_all["bloc_u"] == "DNB_FINAL")].copy()
rank_dnb_global, n_dnb = build_rank_pivot_with_total(dnb_net_global, group_col=None)

# =========================================================
# MODIF KPIs: sous chaque st.metric, on ajoute le mini tableau rang/total
# =========================================================

c1, c2 = st.columns(2)
with c1:
    # BAC global
    s_bac = series_23_25(df_e[df_e["examen_u"] == "BAC"])
    v, d = current_and_delta(s_bac)
    st.metric(
        "Moyenne BAC",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta=None if pd.isna(d) else f"{d:+.2f}",
        border=True,
    )

    # Rang BAC / total
    rank_row_bac = get_global_rank_row(rank_bac_global, n_bac, etab)
    st.caption("Rang (réseau) / total établissements")
    st.dataframe(rank_row_bac, hide_index=True, use_container_width=True, height=70)

with c2:
    # DNB global = DNB_FINAL uniquement
    dnb_mask = (df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")
    s_dnb = series_23_25(df_e[dnb_mask])
    v, d = current_and_delta(s_dnb)
    st.metric(
        "Moyenne DNB (FINAL)",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta=None if pd.isna(d) else f"{d:+.2f}",
        border=True,
    )

    # Rang DNB / total
    rank_row_dnb = get_global_rank_row(rank_dnb_global, n_dnb, etab)
    st.caption("Rang (réseau) / total établissements")
    st.dataframe(rank_row_dnb, hide_index=True, use_container_width=True, height=70)


c1, c2 = st.columns(2)

with c1:
    s_bac = series_23_25(df_e[df_e["examen_u"] == "BAC"])
    v, d = current_and_delta(s_bac)
    st.metric(
        "Moyenne BAC",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta=None if pd.isna(d) else f"{d:+.2f}",
        border=True,
    )

with c2:
    dnb_mask = (df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")
    s_dnb = series_23_25(df_e[dnb_mask])
    v, d = current_and_delta(s_dnb)
    st.metric(
        "Moyenne DNB (FINAL)",
        value="—" if pd.isna(v) else f"{v:.2f}",
        delta=None if pd.isna(d) else f"{d:+.2f}",
        border=True,
    )

st.divider()

# =========================================================
# TABS
# =========================================================
tab_bac, tab_dnb, tab_spe = st.tabs(
    ["BAC — Épreuves finales", "DNB — Épreuves finales", "BAC — Spécialités"]
)

# =========================================================
# TAB 1 — BAC Épreuves finales (cartes par bloc_epreuve)
# =========================================================
with tab_bac:
    bac_etab = df_e[df_e["examen_u"] == "BAC"].copy()
    bac_net  = df_all[df_all["examen_u"] == "BAC"].copy()

    # Pivot rang réseau sur bloc_epreuve
    bac_net_nonnull = bac_net.dropna(subset=["bloc_epreuve"]).copy()
    rank_pivot_bac_bloc = build_rank_pivot(bac_net_nonnull, group_col="bloc_epreuve")

    target_blocs = [
        ("EDS", {"EDS"}),
        ("EAF", {"EAF"}),
        ("Grand Oral", {"GO", "GRAND ORAL"}),
        ("Philosophie", {"PHILO", "PHILOSOPHIE"}),
    ]

    card_items = []
    for label, accepted in target_blocs:
        m = bac_etab["bloc_u"].isin(accepted)
        s = series_23_25(bac_etab[m])

        # Ne pas afficher la carte si aucune donnée (comportement existant)
        if bac_etab[m].empty:
            continue

        # Valeur "réelle" de bloc_epreuve pour indexer le pivot
        bloc_value = bac_etab.loc[m, "bloc_epreuve"].mode().iloc[0]
        rank_df = get_rank_row(rank_pivot_bac_bloc, "bloc_epreuve", bloc_value, etab)

        card_items.append((label, s, rank_df))

    render_cards_grid(card_items, cols=4)

# =========================================================
# TAB 2 — DNB Épreuves finales (cartes par matière = epreuve)
# =========================================================
with tab_dnb:
    dnb_etab = df_e[(df_e["examen_u"] == "DNB") & (df_e["bloc_u"] == "DNB_FINAL")].copy()
    dnb_net  = df_all[(df_all["examen_u"] == "DNB") & (df_all["bloc_u"] == "DNB_FINAL")].copy()

    # Exclusions DNB sur réseau + établissement
    dnb_etab["ep_u"] = norm_series(dnb_etab["epreuve"])
    dnb_net["ep_u"]  = norm_series(dnb_net["epreuve"])

    dnb_etab = dnb_etab[~dnb_etab["ep_u"].isin(DNB_EPREUVE_EXCLUDE)].copy()
    dnb_net  = dnb_net[~dnb_net["ep_u"].isin(DNB_EPREUVE_EXCLUDE)].copy()

    if dnb_etab.empty:
        pass
    else:
        rank_pivot_dnb = build_rank_pivot(dnb_net.dropna(subset=["epreuve"]), group_col="epreuve")

        card_items = []
        for epr in sorted(dnb_etab["epreuve"].dropna().unique().tolist()):
            s = series_23_25(dnb_etab[dnb_etab["epreuve"] == epr])
            rank_df = get_rank_row(rank_pivot_dnb, "epreuve", epr, etab)
            card_items.append((str(epr), s, rank_df))

        render_cards_grid(card_items, cols=5)

# =========================================================
# TAB 3 — BAC Spécialités (cartes par spécialité = epreuve dans EDS)
# =========================================================
with tab_spe:
    bac_etab = df_e[df_e["examen_u"] == "BAC"].copy()
    bac_net  = df_all[df_all["examen_u"] == "BAC"].copy()

    spe_etab = bac_etab[bac_etab["bloc_u"] == "EDS"].copy()
    spe_net  = bac_net[bac_net["bloc_u"] == "EDS"].copy()

    if spe_etab.empty:
        pass
    else:
        rank_pivot_spe = build_rank_pivot(spe_net.dropna(subset=["epreuve"]), group_col="epreuve")

        card_items = []
        for epr in sorted(spe_etab["epreuve"].dropna().unique().tolist()):
            s = series_23_25(spe_etab[spe_etab["epreuve"] == epr])
            rank_df = get_rank_row(rank_pivot_spe, "epreuve", epr, etab)
            card_items.append((str(epr), s, rank_df))

        render_cards_grid(card_items, cols=4)
