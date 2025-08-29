import os
import glob
import pandas as pd
import numpy as np
import tables
import hdf5_getters as GETTERS
from tqdm import tqdm

# Ruta al subset del Million Song Dataset
DATASET_PATH = "MillionSongSubset/"

# Buscar todos los archivos .h5
files = glob.glob(os.path.join(DATASET_PATH, '**/*.h5'), recursive=True)
print(f"Archivos encontrados: {len(files)}")

data = []

for f in tqdm(files, desc="Procesando canciones"):
    try:
        h5 = tables.open_file(f, 'r')

        # Metadata básica
        track_id = GETTERS.get_track_id(h5).decode("utf-8")
        title = GETTERS.get_title(h5).decode("utf-8")
        artist = GETTERS.get_artist_name(h5).decode("utf-8")
        release = GETTERS.get_release(h5).decode("utf-8")
        year = GETTERS.get_year(h5)

        # Features acústicas
        tempo = GETTERS.get_tempo(h5)
        key = GETTERS.get_key(h5)
        mode = GETTERS.get_mode(h5)
        ts = GETTERS.get_time_signature(h5)
        loudness = GETTERS.get_loudness(h5)
        duration = GETTERS.get_duration(h5)
        danceability = GETTERS.get_danceability(h5)
        energy = GETTERS.get_energy(h5)

        # Popularidad
        artist_hotttnesss = GETTERS.get_artist_hotttnesss(h5)
        song_hotttnesss = GETTERS.get_song_hotttnesss(h5)

        # Promedio segments_timbre y segments_pitches (12 dimensiones cada uno)
        segments_timbre = GETTERS.get_segments_timbre(h5)
        segments_pitches = GETTERS.get_segments_pitches(h5)

        timbre_avg = np.mean(segments_timbre, axis=0) if segments_timbre.size else np.zeros(12)
        pitches_avg = np.mean(segments_pitches, axis=0) if segments_pitches.size else np.zeros(12)

        # Guardar todo en una fila
        row = [
            track_id, title, artist, release, year,
            tempo, key, mode, ts, loudness,
            duration, danceability, energy,
            artist_hotttnesss, song_hotttnesss
        ] + timbre_avg.tolist() + pitches_avg.tolist()

        data.append(row)
        h5.close()

    except Exception as e:
        print(f"Error procesando {f}: {e}")

# Columnas
timbre_cols = [f"timbre_{i}" for i in range(12)]
pitches_cols = [f"pitch_{i}" for i in range(12)]
columns = [
    "track_id", "title", "artist_name", "release", "year",
    "tempo", "key", "mode", "time_signature", "loudness",
    "duration", "danceability", "energy",
    "artist_hotttnesss", "song_hotttnesss"
] + timbre_cols + pitches_cols

# Crear DataFrame
df = pd.DataFrame(data, columns=columns)

# Guardar CSV
output_file = "msd_subset_features_full.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ CSV guardado como {output_file} con {len(df)} canciones.")
