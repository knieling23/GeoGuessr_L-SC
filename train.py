import os
from data_loader import load_data
from build_model import build_cnn_model
from sklearn.model_selection import train_test_split

# Daten laden
X, y = load_data(
    csv_path='../dataset/coords.csv',
    images_dir='../dataset',
    img_size=(500, 500)
)

# Splitten
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell bauen
model = build_cnn_model(input_shape=(224, 224, 3))

# Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Modell speichern
os.makedirs('../models', exist_ok=True)
model.save('../models/geolocation_cnn.h5')
