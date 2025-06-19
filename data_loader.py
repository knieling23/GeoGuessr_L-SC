import pandas as pd
import numpy as np
from PIL import Image
import os

def load_data(csv_path, images_dir, img_size=(500, 500), max_samples=None):
    df = pd.read_csv(csv_path, dtype={'ID': str})
    images = []
    labels = []

    for idx, row in df.iterrows():
        img_filename = f"{row['ID']}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB').resize(img_size)
            images.append(np.array(img))
            labels.append([row['Latitude'], row['Longitude']])
        if max_samples and len(images) >= max_samples:
            break

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.float32)
    return X, y

# Hier gibst du die richtigen Pfade an:
csv_path = 'dataset/coords.csv'
images_dir = 'dataset'

print(f"CSV-Pfad: {csv_path}")
print(f"Bilder-Verzeichnis: {images_dir}")
