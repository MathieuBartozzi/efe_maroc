import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(file_key: str):
    """
    Charge un fichier CSV stocké sur Google Drive à partir de la clé
    définie dans .streamlit/secrets.toml
    """
    try:
        file_id = st.secrets["google"][file_key]
        url = f"https://drive.google.com/uc?id={file_id}"
        df = pd.read_csv(url)
                # Conversion automatique des colonnes latitude/longitude si elles existent
        for col in df.columns:
            if col.lower() in ["lat","long"]:
                df[col] = (
                    df[col]
                    .astype(str)              # s'assure qu'on manipule du texte
                    .str.replace(",", ".", regex=False)  # convertit les virgules en points
                    .astype(float)            # conversion en float
                )

        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier {file_key} : {e}")
        return pd.DataFrame()
