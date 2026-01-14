import streamlit as st
from typing import Iterable

# def _normalize_email(email: str) -> str:
#     return (email or "").strip().lower()

# def _normalize_email(email: str) -> str:
#     return (email or "").strip().lower()

# def authenticate(*, auth_profile: str = "auth_all"):
#     """
#     Authentification Streamlit configurable par profil de secrets.

#     auth_profile : nom de la section dans secrets.toml
#       ex: "auth_all", "auth_selected_email"
#     """

#     # Déjà connecté
#     if st.session_state.get("auth_ok"):
#         return st.session_state["username_friendly"]

#     st.header("🔒 Connexion")

#     with st.form("login_form"):
#         email = st.text_input("Adresse e-mail")
#         password = st.text_input("Mot de passe", type="password")
#         submitted = st.form_submit_button("Se connecter")

#     if not submitted:
#         st.stop()

#     # Lecture du profil
#     try:
#         config = st.secrets[auth_profile]
#     except KeyError:
#         st.error(f"Profil d'authentification introuvable : [{auth_profile}]")
#         st.stop()

#     # Mot de passe
#     shared_password = config.get("password")
#     if not shared_password:
#         st.error("Mot de passe manquant dans les secrets.")
#         st.stop()

#     allowed_domain = config.get("allowed_domain")
#     allowed_emails = config.get("allowed_emails")

#     if not allowed_domain and not allowed_emails:
#         st.error("Configuration invalide : aucun mode d'autorisation défini.")
#         st.stop()

#     clean_email = _normalize_email(email)

#     # Contrôle d'accès
#     domain_ok = (
#         isinstance(allowed_domain, str)
#         and clean_email.endswith(allowed_domain.strip().lower())
#     )

#     list_ok = (
#         isinstance(allowed_emails, Iterable)
#         and clean_email in {_normalize_email(e) for e in allowed_emails}
#     )

#     if not (domain_ok or list_ok):
#         st.error("Accès refusé : adresse non autorisée.")
#         st.stop()

#     # Vérification mot de passe
#     if password != shared_password:
#         st.error("Mot de passe incorrect.")
#         st.stop()

#     # Succès
#     friendly_name = clean_email.split("@")[0].replace(".", " ").title()

#     st.session_state["auth_ok"] = True
#     st.session_state["user_email"] = clean_email
#     st.session_state["username_friendly"] = friendly_name
#     st.rerun()



# def logout():
#     for key in ["auth_ok", "user_email", "username_friendly", "show_welcome"]:
#         st.session_state.pop(key, None)
#     st.success("Vous avez été déconnecté.")
#     st.rerun()

# =========================================================
# AUTH
# =========================================================
import streamlit as st
from typing import Iterable

def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def authenticate(*, auth_profile: str = "auth_all"):
    # Déjà connecté
    if st.session_state.get("auth_ok"):
        st.session_state.setdefault("can_view_reseau_rank", False)
        return st.session_state.get("username_friendly", "")

    st.header("Connexion")

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

    # Contrôle d'accès (login)
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

    # -------- DROIT RESTREINT : accès au bloc "Classement réseau" --------
    restricted_emails = config.get("restricted_emails")
    can_view_reseau_rank = (
        isinstance(restricted_emails, Iterable)
        and clean_email in {_normalize_email(e) for e in restricted_emails}
    )

    friendly_name = clean_email.split("@")[0].replace(".", " ").title()

    st.session_state["auth_ok"] = True
    st.session_state["user_email"] = clean_email
    st.session_state["username_friendly"] = friendly_name
    st.session_state["can_view_reseau_rank"] = bool(can_view_reseau_rank)

    st.rerun()

def logout():
    for key in ["auth_ok", "user_email", "username_friendly", "can_view_reseau_rank", "show_welcome"]:
        st.session_state.pop(key, None)
    st.success("Vous avez été déconnecté.")
    st.rerun()

