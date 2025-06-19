import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Modell laden
model = load_model('models/geolocation_cnn.keras')

# Ordner mit Screenshots
input_folder = 'screenshots'

# Unterstützte Bildformate
valid_exts = ('.jpg', '.jpeg', '.png')

# Alle Bilddateien im Ordner finden und sortieren
image_files = sorted([f for f in os.listdir(input_folder)
                      if f.lower().endswith(valid_exts)])

if not image_files:
    print("Keine Bilder im Ordner 'screenshots' gefunden.")
    exit()

# Für jedes Bild Vorhersage machen
for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    try:
        img = Image.open(img_path).convert('RGB').resize((500, 500))
        X = np.array(img, dtype=np.float32)[None, ...] / 255.0
        pred = model.predict(X)
        lat, lon = pred[0]
        print(f"{filename}: Vorhergesagte Koordinaten -> Breitengrad: {lat:.6f}, Längengrad: {lon:.6f}")
    except Exception as e:
        print(f"Fehler bei {filename}: {e}")
