# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Titre
st.title("Détection d'anomalies avec un Autoencodeur")
st.markdown("Téléchargez un fichier CSV pour analyser les anomalies")

# Charger le modèle et le scaler
@st.cache_resource
def load_artifacts():
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

autoencoder, scaler = load_artifacts()

# Upload de fichier
uploaded_file = st.file_uploader("Importer un fichier CSV", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    # Mise à l'échelle
    data_scaled = scaler.transform(data)

    # Reconstruction
    reconstructed = autoencoder.predict(data_scaled)
    reconstruction_error = np.mean(np.square(data_scaled - reconstructed), axis=1)

    # Seuil basé sur les données d'entraînement (à adapter selon ton seuil optimal)
    threshold = np.percentile(reconstruction_error, 95)

    # Prédiction
    predictions = (reconstruction_error > threshold).astype(int)  # 1 = anomalie

    # Résultat final
    results_df = data.copy()
    results_df['Reconstruction_Error'] = reconstruction_error
    results_df['Anomalie'] = predictions

    st.subheader("Résultats de la détection")
    st.write(f"Seuil utilisé : {threshold:.4f}")
    st.dataframe(results_df)

    # Téléchargement
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger les résultats", csv, "anomalies_resultats.csv", "text/csv")
