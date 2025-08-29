import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import openpyxl

df = pd.read_excel("train_v2.xlsx")
df.fillna(0, inplace=True)

features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms", "time_signature", "popularity"
]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=13, random_state=42)
X_reduced = pca.fit_transform(X_scaled)
knn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
knn.fit(X_reduced)

def recomendar_canciones(song_title, n_recommendations=10):
    if song_title not in df['track_name'].values:
        print(f"'{song_title}' no está en el dataset.")
        return [] 
    idx = df.index[df['track_name'] == song_title][0]
    distances, indices = knn.kneighbors([X_reduced[idx]], n_neighbors=n_recommendations+1)    
    recommended_indices = indices.flatten()[1:]
    recommended_songs = df.iloc[recommended_indices][["track_name", "artists", "track_genre"]]
    return recommended_songs

st.title("Music Recommender App")
song_list = sorted(df["track_name"].dropna().astype(str).unique())
song_choice = st.selectbox("Elige una canción:", song_list)

n_recs = st.slider("Número de recomendaciones:", 1, 20, 5)

if st.button("Recomendar"):
    recs = recomendar_canciones(song_choice, n_recs)
    if recs is None:
        st.error("❌ Canción no encontrada en el dataset.")
    else:
        st.subheader(f"🔎 Recomendaciones para '{song_choice}':")
        st.dataframe(recs.reset_index(drop=True))



