import h5py
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Absolute paths
h5_path = 'Cancer_Dataset_Files/all_mias_scans.h5'
info_path = 'Cancer_Dataset_Files/Info.txt'

print("Current working directory:", os.getcwd())
print("Checking info_path existence:", os.path.exists(info_path))
print("Checking h5_path existence:", os.path.exists(h5_path))
print("Absolute info_path:", os.path.abspath(info_path))
print("Absolute h5_path:", os.path.abspath(h5_path))

def load_labels():
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Could not find {info_path}. Please check the file path.")
    df = pd.read_csv(info_path, sep=r'\s+', comment='#',
                     names=['ref', 'bg', 'abnorm_class', 'severity', 'x', 'y', 'radius'],
                     skiprows=1,
                     on_bad_lines='skip')
    df = df.dropna(subset=['ref'])
    df['label'] = df['abnorm_class'].apply(lambda x: 0 if pd.isna(x) or x == '' or x == 'NORM' else 1)
    print("Sample labels:", df[['ref', 'abnorm_class', 'label']].head(10).to_string())
    return df[['ref', 'label']]

def load_images(df_labels, target_size=(224, 224)):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Could not find {h5_path}. Please check the file path.")
    images = []
    labels = []
    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj):
            print(f"{name}: type={type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset shape: {obj.shape}, dtype: {obj.dtype}")
        f.visititems(print_structure)

        if 'scan' in f:
            scan_data = np.array(f['scan'])
            for i in range(min(len(df_labels), len(scan_data))):
                img = cv2.resize(scan_data[i], target_size)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)
                images.append(img)
                labels.append(df_labels['label'].iloc[i])
                #print(f"Loaded image {i} for ref {df_labels['ref'].iloc[i]}, shape {img.shape}")
    if not images:
        print("Warning: No images loaded. Check the HDF5 structure output above.")
    return np.array(images), np.array(labels)

def preprocess_data():
    df_labels = load_labels()
    print(f"Loaded {len(df_labels)} labels. Normals: {sum(df_labels['label'] == 0)}, Abnormals: {sum(df_labels['label'] == 1)}")
    X, y = load_images(df_labels)
    print(f"Loaded {X.shape[0]} images with shape {X.shape[1:]}\n")
    if X.shape[0] == 0:
        raise ValueError("No images loaded. Please check the HDF5 file structure.")
    y = to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1))
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,  # Reduced augmentation
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    datagen.fit(X_train)
    return X_train, X_test, y_train, y_test, datagen