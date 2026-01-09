# import streamlit as st

# def authenticate():
#     """
#     Vérifie l'identité de l'utilisateur via un formulaire simple.
#     Accepte TOUTE adresse se terminant par @mlfmonde.org
#     Utilise st.secrets["auth"]["password"] pour le mot de passe partagé.
#     Retourne le prénom/nom formaté depuis l'email si connexion réussie.
#     """

#     # Si déjà connecté, on ne redemande pas
#     if st.session_state.get("auth_ok"):
#         return st.session_state["username_friendly"]

#     st.header("🔒 Connexion")

#     # Formulaire de connexion
#     with st.form("login_form"):
#         email = st.text_input("Adresse e-mail (@mlfmonde.org)")
#         password = st.text_input("Mot de passe", type="password")
#         submitted = st.form_submit_button("Se connecter")

#     if not submitted:
#         st.stop()  # on arrête le script tant que l'utilisateur n'a pas cliqué

#     # Récupération du mot de passe partagé depuis secrets.toml
#     # Note : On n'a plus besoin de la liste ["users"]
#     try:
#         shared_password = st.secrets["auth"]["password"]
#     except KeyError:
#         st.error("Erreur de configuration : Mot de passe introuvable dans les secrets.")
#         st.stop()

#     # 1. Vérification du DOMAINE de l'email
#     # .strip() enlève les espaces accidentels, .lower() gère les majuscules
#     clean_email = email.strip().lower()

#     if not clean_email.endswith("@mlfmonde.org"):
#         st.error("Accès restreint aux adresses @mlfmonde.org uniquement.")
#         st.stop()

#     # 2. Vérification du mot de passe
#     if password != shared_password:
#         st.error("Mot de passe incorrect.")
#         st.stop()

#     # ✅ Authentification réussie

#     # On génère un "Nom Friendly" à partir de l'email
#     # ex: jean.dupont@mlfmonde.org -> Jean Dupont
#     user_part = clean_email.split("@")[0]
#     friendly_name = user_part.replace(".", " ").title()

#     st.session_state["auth_ok"] = True
#     st.session_state["user_email"] = clean_email
#     st.session_state["username_friendly"] = friendly_name
#     st.session_state["show_welcome"] = True
#     st.rerun()

#     return friendly_name


# def logout():
#     """
#     Déconnecte l'utilisateur en réinitialisant l'état de session.
#     """
#     for key in ["auth_ok", "user_email", "username_friendly", "show_welcome"]:
#         if key in st.session_state:
#             del st.session_state[key]
#     st.success("Vous avez été déconnecté.")
#     st.rerun() # Utiliser rerun() pour rafraîchir la page immédiatement

import streamlit as st
from typing import Iterable

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def authenticate(*, auth_profile: str = "auth_all"):
    """
    Authentification Streamlit configurable par profil de secrets.

    auth_profile : nom de la section dans secrets.toml
      ex: "auth_all", "auth_selected_email"
    """

    # Déjà connecté
    if st.session_state.get("auth_ok"):
        return st.session_state["username_friendly"]

    st.header("🔒 Connexion")

    with st.form("login_form"):
        email = st.text_input("Adresse e-mail")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if not submitted:
        st.stop()

    # Lecture du profil
    try:
        config = st.secrets[auth_profile]
    except KeyError:
        st.error(f"Profil d'authentification introuvable : [{auth_profile}]")
        st.stop()

    # Mot de passe
    shared_password = config.get("password")
    if not shared_password:
        st.error("Mot de passe manquant dans les secrets.")
        st.stop()

    allowed_domain = config.get("allowed_domain")
    allowed_emails = config.get("allowed_emails")

    if not allowed_domain and not allowed_emails:
        st.error("Configuration invalide : aucun mode d'autorisation défini.")
        st.stop()

    clean_email = _normalize_email(email)

    # Contrôle d'accès
    domain_ok = (
        isinstance(allowed_domain, str)
        and clean_email.endswith(allowed_domain.strip().lower())
    )

    list_ok = (
        isinstance(allowed_emails, Iterable)
        and clean_email in {_normalize_email(e) for e in allowed_emails}
    )

    if not (domain_ok or list_ok):
        st.error("Accès refusé : adresse non autorisée.")
        st.stop()

    # Vérification mot de passe
    if password != shared_password:
        st.error("Mot de passe incorrect.")
        st.stop()

    # Succès
    friendly_name = clean_email.split("@")[0].replace(".", " ").title()

    st.session_state["auth_ok"] = True
    st.session_state["user_email"] = clean_email
    st.session_state["username_friendly"] = friendly_name
    st.rerun()



def logout():
    for key in ["auth_ok", "user_email", "username_friendly", "show_welcome"]:
        st.session_state.pop(key, None)
    st.success("Vous avez été déconnecté.")
    st.rerun()
