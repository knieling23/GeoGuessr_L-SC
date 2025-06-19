import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os

class GeoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, images_dir, batch_size=32, img_size=(500, 500), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_indices]
        images = []
        labels = []
        for _, row in batch.iterrows():
            img_filename = f"{row['ID']}.jpg"
            img_path = os.path.join(self.images_dir, img_filename)
            img = Image.open(img_path).convert('RGB').resize(self.img_size)
            images.append(np.array(img) / 255.0)
            labels.append([row['Latitude'], row['Longitude']])
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
