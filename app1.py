import streamlit as st
import numpy as np
import pandas as pd
from implicit.datasets.lastfm import get_lastfm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

st.write("📥 Cargando dataset de Last.fm...")
artists, users, artist_user_plays = get_lastfm()

artist_user_norm = normalize(artist_user_plays)

sim_matrix = cosine_similarity(artist_user_norm)

def recomendar_artistas(artist_name, n=5):
    if artist_name not in artists:
        return f"❌ El artista '{artist_name}' no está en el dataset."
    
    idx = artists.index(artist_name)
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i for i, _ in sim_scores[1:n+1]]
    return [artists[i] for i in top_indices]

st.title("🎵 Recomendador de Artistas - Last.fm")

artist_choice = st.selectbox("Elige un artista:", sorted(artists))

n_recs = st.slider("Número de recomendaciones:", 1, 20, 5)

if st.button("Recomendar"):
    recs = recomendar_artistas(artist_choice, n_recs)
    if isinstance(recs, str): 
        st.error(recs)
    else:
        st.subheader(f"🔎 Recomendaciones para '{artist_choice}':")
        st.write(recs)
