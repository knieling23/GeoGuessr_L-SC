import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_data

# Daten laden
X_test, y_test = load_data(
    csv_path='../dataset/coords.csv',
    images_dir='../dataset',
    img_size=(500, 500),
    max_samples=1000  
)

# Modell laden
model = load_model('../models/geolocation_cnn.h5')

# Vorhersage
y_pred = model.predict(X_test)

# Fehler berechnen (mittlerer euklidischer Abstand)
distances = np.sqrt(np.sum((y_pred - y_test) ** 2, axis=1))
print(f"Mittlerer Fehler (Grad): {np.mean(distances)}")
