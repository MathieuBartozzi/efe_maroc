import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# -------------------------------------------------
from utils.loader import load_data
from utils.auth import authenticate, logout

# --- Configuration de la page ---
st.set_page_config(
    page_title="MLF - Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="expanded",
)



# --- Correction du logo ---
# On calcule le chemin absolu basé sur l'emplacement de app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo_mlfmonde.png")

#On vérifie que le fichier existe (optionnel, pour le debug)
if not os.path.exists(logo_path):
    st.error(f"Image introuvable : {logo_path}")


# # --- Authentification utilisateur ---
# user = authenticate()

# if st.session_state.get("show_welcome", False):
#     st.success(f"Bienvenue, {user} ! 🎉")
#     st.session_state["show_welcome"] = False

# --- Chargement des données ---
with st.spinner("Chargement des données…"):
    df = load_data("result_id")


# Sauvegarde en mémoire pour toute la session
st.session_state["df"] = df

# --- Navigation multipage ---
pages = [
    st.Page("app_pages/1_vue_reseau.py", title="RÉSEAU", icon=":material/globe:"),
    st.Page("app_pages/2_vue_etablissement.py", title="ÉTABLISSEMENT", icon=":material/school:"),
    # st.Page("app_pages/3_exploration_avancee.py", title="EXPLORATION", icon=":material/chat:"),
]

pg = st.navigation(pages, position="top")

st.logo(logo_path, size="large")

# --- Exécuter la page active ---
pg.run()




# --- Déconnexion ---
if st.sidebar.button("Se déconnecter"):
    logout()
