import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from sklearn.model_selection import train_test_split
from build_model import build_cnn_model
from geo_data_generator import GeoDataGenerator
import time

# Debugging-Ausgaben
print("TensorFlow Version:", os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'Nicht gesetzt'))
print("Aktuelles Verzeichnis:", os.getcwd())
print("Existiert coords.csv:", os.path.exists('dataset/coords.csv'))

# >>> Testmodus aktivieren: True = nur kleiner Teil, False = alles
TESTMODE = True
TEST_SAMPLES = 100  # Anzahl der Beispiele im Testmodus

# 1. Daten laden und splitten
df = pd.read_csv('dataset/coords.csv', dtype={'ID': str})

if TESTMODE:
    df = df.sample(n=min(TEST_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

# Split: 70% Training, 15% Validation, 15% Test
df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

print(f"Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

# 2. Generatoren anlegen
train_gen = GeoDataGenerator(df_train, images_dir='dataset', batch_size=8, img_size=(500, 500))
val_gen   = GeoDataGenerator(df_val,   images_dir='dataset', batch_size=8, img_size=(500, 500), shuffle=False)
# Test-Generator wird nur für evaluate.py benötigt

# 3. Modell bauen
model = build_cnn_model(input_shape=(500, 500, 3))

start = time.time()
# 4. Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2  # Wenige Epochen für schnellen Test
)

print(f"Laufzeit: {time.time() - start:.2f} Sekunden")

# 5. Modell speichern
os.makedirs('models', exist_ok=True)
model.save('models/geolocation_cnn.keras')
