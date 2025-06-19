import pandas as pd
from geo_data_generator import GeoDataGenerator
from build_model import build_cnn_model
from sklearn.model_selection import train_test_split

# CSV laden
df = pd.read_csv('dataset/coords.csv', dtype={'ID': str})

# Split in Training und Validation
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

# Generatoren erstellen
train_gen = GeoDataGenerator(df_train, images_dir='dataset', batch_size=32, img_size=(500, 500))
val_gen = GeoDataGenerator(df_val, images_dir='dataset', batch_size=32, img_size=(500, 500), shuffle=False)

# Modell bauen
model = build_cnn_model(input_shape=(500, 500, 3))

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Modell speichern
import os
os.makedirs('models', exist_ok=True)
model.save('models/geolocation_cnn.h5')
