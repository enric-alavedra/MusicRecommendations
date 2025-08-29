import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import implicit

@st.cache_data
def load_data():
    df = pd.read_csv("usersha1-artmbid-artname-plays.tsv", sep="\t", header=None,
                     names=["user_id", "artist_mbid", "artist_name", "plays"])
    df.fillna("", inplace=True)
    
    user_activity = df.groupby("user_id")["plays"].sum()
    active_users = user_activity[user_activity > 50].index
    df = df[df["user_id"].isin(active_users)]

    user_enc = LabelEncoder()
    artist_enc = LabelEncoder()
    df["user_idx"] = user_enc.fit_transform(df["user_id"])
    df["artist_idx"] = artist_enc.fit_transform(df["artist_name"])

    plays_matrix = csr_matrix((df["plays"], (df["user_idx"], df["artist_idx"])))
    return df, user_enc, artist_enc, plays_matrix
  
df, user_enc, artist_enc, plays_matrix = load_data()

@st.cache_resource
def train_model(plays_matrix):
    model = implicit.als.AlternatingLeastSquares(
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42
    )
    model.fit(plays_matrix.T) 
    return model

model = train_model(plays_matrix)

def recomendar_artistas(artist_name, n=5):
    if artist_name not in artist_enc.classes_:
        return pd.DataFrame({"Error": [f"❌ El artista '{artist_name}' no está en el dataset"]})
    
    artist_idx = artist_enc.transform([artist_name])[0]
    similar_artists = model.similar_items(artist_idx, N=n+1)
    
    results = []
    for idx, score in similar_artists[1:]:  
        results.append({
            "artist_name": artist_enc.inverse_transform([idx])[0],
            "score": score
        })
    return pd.DataFrame(results)


st.title("🎵 Last.fm Artist Recommender")

artist_list = sorted(df["artist_name"].unique())
artist_choice = st.selectbox("Elige un artista:", artist_list)

n_recs = st.slider("Número de artistas recomendados:", 1, 20, 5)

if st.button("Recomendar"):
    recs = recomendar_artistas(artist_choice, n_recs)
    if "Error" in recs.columns:
        st.error(recs["Error"].iloc[0])
    else:
        st.subheader(f"🔎 Artistas similares a '{artist_choice}':")
        st.dataframe(recs.reset_index(drop=True))
