# import streamlit as st
# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr
# import sys, os

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from fonctions_viz import *

# # =====================================================
# # 0. Chargement des données
# # =====================================================
# df = st.session_state.get("df")
# df_coordo = st.session_state.get("df_coordo")

# if df is None or df.empty:
#     st.warning("Aucune donnée disponible. Ouvrez d'abord la page Home.")
#     st.stop()

# df["Valeur"] = df["Valeur"] * 100

# # =====================================================
# # Fonctions évolution
# # =====================================================

# def evolution_slope(g):
#     g = g.sort_values("niveau_code")
#     x = g["niveau_code"].values
#     y = g["Valeur"].values
#     if len(x) < 2:
#         return np.nan
#     return np.polyfit(x, y, 1)[0]

# def evolution_spearman(g):
#     g = g.sort_values("niveau_code")
#     x = g["niveau_code"].values
#     y = g["Valeur"].values
#     return np.nan if len(x) < 2 else spearmanr(x, y)[0]

# def delta_first_last(g):
#     g = g.sort_values("niveau_code")
#     return np.nan if g.shape[0] < 2 else g["Valeur"].iloc[-1] - g["Valeur"].iloc[0]



# # =====================================================
# # Interface Principale
# # =====================================================
# st.subheader("Exploration avancée des compétences")

# onglets = st.tabs([
#     "Statistiques globales",
#     "Compétences discriminantes",
#     "Évolution CP → CM2",
#     "Grille de lecture",
# ])

# # =====================================================
# # ONGLET 1 : Statistiques globales
# # =====================================================
# with onglets[0]:

#     with st.container(border=True):
#         st.subheader("Statistiques globales des compétences")


#         col1, col2, col3 = st.columns(3)

#         # COLONNE 1 : distribution FR/MATH
#         with col1:
#             # fig = px.histogram(df, x="Valeur", color="Matière", barmode="overlay", nbins=30)
#             # st.plotly_chart(fig, width='stretch')
#             plot_distribution_competences(df, palette)

#         # COLONNE 2 : distribution par matière sélectionnée
#         with col2:
#             vue_top_bottom_matiere(df, "Français", n=7)

#         with col3:
#             vue_top_bottom_matiere(df, "Mathématiques", n=7)

# # =====================================================
# # ONGLET 2 : Compétences discriminantes
# # =====================================================
# with onglets[1]:

#     with st.container(border=True) :
#         st.subheader("Compétences discriminantes ")

#         st.markdown("""
#                     > Une compétence est dite « discriminante » lorsqu'elle permet de distinguer clairement les établissements : les écarts de performance y sont importants. Ici, la discrimination est estimée via la dispersion des scores (écart-type). Plus l'écart-type est élevé, plus la compétence différencie les établissements : en dessous de 8 la dispersion est faible, entre 8 et 12 elle devient notable, au-delà de 12 la compétence est fortement discriminante.
#                     """)
#         # --- SLIDER en haut ---
#         col_slider_left, col_slider_empty = st.columns([1, 3])
#         with col_slider_left:
#             seuil_std = st.slider(
#                 "Seuil de discrimination",
#                 min_value=7,
#                 max_value=17,
#                 value=12,
#                 step=1,
#             )

#         col1, col2=st.columns(2)
#         with col1:
#             # st.plotly_chart(plot_swarm_competences(df, palette, seuil_std=seuil_std), width='stretch')
#             plot_swarm_competences(df, palette, seuil_std=seuil_std)


#         with col2:
#             # st.plotly_chart(plot_scatter_dispersion(df, palette, seuil_std=seuil_std), width='stretch')
#             plot_scatter_dispersion(df, palette, seuil_std=seuil_std)


#         df_fr, df_math = list_competences_discriminantes(df, seuil_std=seuil_std)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.dataframe(df_fr, width='stretch')

#         with col2:
#             st.dataframe(df_math, width='stretch')

# # =====================================================
# # ONGLET 3 : Évolution CP → CM2
# # =====================================================



# with onglets[2]:
#         #=====================================================
#     # 1) PRÉPARATION DES DONNÉES
#     # =====================================================

#     # Valeurs moyennes par COMPÉTENCE et par NIVEAU
#     df_reseau = (
#         df.groupby(["Matière", "Domaine", "Compétence", "Niveau"])["Valeur"]
#         .mean()
#         .reset_index()
#     )

#     # Code niveau (CP=0 → CM2=4)
#     df_reseau["niveau_code"] = df_reseau["Niveau"].apply(lambda x: ordre_niveaux.index(x))


#     # =====================================================
#     # 2) EVOLUTION PAR COMPÉTENCE
#     # =====================================================
#     df_evol_comp = (
#         df_reseau.groupby(["Matière", "Domaine", "Compétence"])
#         .apply(lambda g: pd.Series({
#             "slope": evolution_slope(g),
#             "spearman": evolution_spearman(g),
#             "delta": delta_first_last(g),
#             "nb_niveaux": g["niveau_code"].nunique()
#         }),include_groups=False)
#         .reset_index()
#     )

#     df_evol_comp_plot = df_evol_comp[df_evol_comp["nb_niveaux"] >= 2]


#     # =====================================================
#     # 3) EVOLUTION PAR DOMAINE
#     # =====================================================

#     df_reseau_dom = (
#         df.groupby(["Matière", "Domaine", "Niveau"])["Valeur"]
#         .mean()
#         .reset_index()
#     )
#     df_reseau_dom["niveau_code"] = df_reseau_dom["Niveau"].apply(lambda x: ordre_niveaux.index(x))

#     df_evol_dom = (
#         df_reseau_dom.groupby(["Matière", "Domaine"])
#         .apply(lambda g: pd.Series({
#             "slope": evolution_slope(g),
#             "spearman": evolution_spearman(g),
#             "delta": delta_first_last(g),
#             "nb_niveaux": g["niveau_code"].nunique()
#         }),include_groups=False)
#         .reset_index()
#     )

#     df_evol_dom_plot = df_evol_dom[df_evol_dom["nb_niveaux"] >= 2]

#     st.subheader("Évolution des compétences CP → CM2")

#     col1, col2= st.columns([2,1])


#     with col1:
#         st.markdown("""
#   > La **pente** mesure l'intensité de la progression :
#   > - Positive → progression
#   > - Négative → régression
#   > - Faible → stagnation

#   > La **corrélation de Spearman** mesure la régularité de la progression
#   > - Proche de 1 → progression régulière
#   > - Proche de 0 → évolution instable

#     """)
#     with col2:
#         st.info(
#             """
#             **À repérer** :

#         > **Régularité faible** -> variations fortes entre niveaux

#         > **Pente positive** ->  point d'appui

#         > **Pente faible** + **régularité faible** -> priorité

#         > **Pente négative** + **régularité élevée** -> alerte

#             """,
#             icon=":material/info:"
#         )





#     # =====================================================
#     # TOP / BOTTOM + GRAPHIQUE DOMAINES + BARRES DOMAINES
#     # =====================================================
#     with st.container(border=True):
#         st.subheader("Évolutions par compétences et par domaines")

#         col1, col2 = st.columns(2)

#         # === COMPÉTENCES ===
#         with col1:
#             afficher_top_bottom_evolutions(df_evol_comp_plot)

#         # === DOMAINES : BAR CHART ===
#         with col2:
#             afficher_bars_progression_regularity(df_evol_dom_plot, palette)





#     # =====================================================
#     # COURBES EN GRILLE PAR NIVEAU
#     # =====================================================
#     with st.container(border=True):
#         st.subheader("Évolutions par niveau")
#         nb = st.pills("Sélectionner le nombre de niveaux", [2,3,4,5],default=2)
#         afficher_courbes_en_grille(df_reseau, df_evol_comp, nb_niveaux=nb, n_cols=4)

# # =====================================================
# # ONGLET 4 : Corrélations améliorées
# # =====================================================
# with onglets[3]:
#     st.markdown("""
# ### Grille de lecture - Exploration avancée des compétences

# Cette page permet d'examiner les compétences en profondeur, non pas à l'échelle d'un établissement, mais à l'échelle des **dynamiques pédagogiques du réseau**. Les trois vues complémentaires aident à comprendre la structure globale des apprentissages du CP au CM2.

# ---

# ### 1. Statistiques globales des compétences

# - Les histogrammes indiquent la **répartition des niveaux de maîtrise** :
#   - forte concentration entre 70-80 % → compétences globalement stabilisées dans le réseau ;
#   - présence de nombreuses compétences < 65 % → fragilités structurelles partagées.
# - Les listes “+ maîtrisées / - maîtrisées” révèlent :
#   - les compétences **généralement robustes** dans le Mlfmonde ;
#   - les compétences **traditionnellement exigeantes** (ex. étude de la langue, automatisation mathématique), pour lesquelles un écart local est souvent normal.
# - Cette vue aide à **contextualiser les difficultés locales** dans une réalité réseau plus large.

# ---

# ### 2. Compétences discriminantes

# - Une compétence est “discriminante” si les écarts entre établissements y sont **très importants**.
# - Le niveau de dispersion (écart-type) est l'indicateur clé :
#   - dispersion < 8 → homogénéité réseau ;
#   - dispersione entre 8-12 → variations significatives ;
#   - dispersion > 12 → compétence fortement différenciante.
# - Les points en zone rouge signalent des compétences :
#   - souvent **trop complexes** ou instables au niveau réseau,
#   - fortement dépendantes des pratiques pédagogiques.
# - Cette vue permet de :
#   - repérer les compétences où un **accompagnement réseau** serait le plus pertinent,
#   - relativiser les écarts d'un établissement lorsque la compétence est **globalement très dispersive**.

# ---

# ### 3. Progression CP → CM2

# - La **pente (slope)** indique l'évolution de la compétence :
#   - positive → progression attendue ;
#   - négative → régression à surveiller ;
#   - faible → stagnation.
# - La **corrélation de Spearman** mesure la cohérence :
#   - élevée → progression régulière et structurée ;
#   - faible → évolution instable ou peu lisible.
# - À repérer en priorité :
#   - les compétences avec **pente négative ou irrégulière** → leviers de continuité pédagogique ;
#   - les compétences avec **pente fortement positive** → appuis pour la formation et la mutualisation.
# - Cette vue met en lumière la **cohérence verticale** réelle des apprentissages du CP au CM2.

# ---

# ### 4. Conclusion

# Cette page donne une vue fine et transversale des compétences à l'échelle du réseau.
# Elle permet de repérer :
# - les compétences structurellement fortes ou fragiles,
# - celles qui génèrent le plus d'écarts entre établissements,
# - celles où la progression CP→CM2 est la plus cohérente ou la plus instable.

# Elle constitue un appui pour **orienter les priorités d'analyse**, les actions de formation et les accompagnements pédagogiques ciblés.




#                 """)
