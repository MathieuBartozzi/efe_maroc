import streamlit as st
import sys, os

# =========================
# MODE MAINTENANCE (tout en haut)
# =========================
MAINTENANCE_MODE = st.secrets["admin"]["MAINTENANCE_MODE"]


if MAINTENANCE_MODE:
    st.set_page_config(
        page_title="Maintenance - MLF Dashboard",
        page_icon="🛠️",
        layout="centered",
    )

    st.markdown(
        """
        <div style="max-width:700px;margin:0 auto;padding-top:80px;text-align:center;">
            <h1>Maintenance en cours</h1>
            <p style="font-size: 1.1rem;">
                Le tableau de bord est temporairement indisponible.
            </p>
            <p style="color: #666;">
                Merci de réessayer un peu plus tard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

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


# --- choisir le mode d'authentification utilisateur ---
# user=authenticate(auth_profile="auth_all")
user=authenticate(auth_profile="auth_selected_email")


if st.session_state.get("show_welcome", False):
    st.success(f"Bienvenue, {user} ! 🎉")
    st.session_state["show_welcome"] = False

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
