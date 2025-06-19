import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_data

# Debugging-Ausgabe
print("TensorFlow Version:", os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'Nicht gesetzt'))
print("Aktuelles Verzeichnis:", os.getcwd())
print("Existiert coords.csv:", os.path.exists('dataset/coords.csv'))

# Daten laden 
X_test, y_test = load_data(
    csv_path='dataset/coords.csv',
    images_dir='dataset',
    img_size=(500, 500),
    max_samples=100  # Optional: nur Teilmenge testen
)

# Modell laden
model = load_model('models/geolocation_cnn.keras')

# Vorhersage
y_pred = model.predict(X_test)

# Fehler berechnen (z.B. mittlerer euklidischer Abstand)
distances = np.sqrt(np.sum((y_pred - y_test) ** 2, axis=1))
print(f"Mittlerer Fehler (Grad): {np.mean(distances)}")
