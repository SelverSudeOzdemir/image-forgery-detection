import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# === VERİ YÜKLEME ===
CASIA_PATH = os.path.expanduser("~/Desktop/CASIA/CASIA2")
IMG_SIZE = (128, 128)
MAX_PER_CLASS = 1000  # MacBook için makul sayı

def load_images(folder, label, max_count):
    images, labels = [], []
    exts = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)][:max_count]
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                from PIL import Image
                img = np.array(Image.open(path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Atlandı: {fname} - {e}")
    return images, labels

print("Au (orijinal) yükleniyor...")
au_imgs, au_labels = load_images(os.path.join(CASIA_PATH, "Au"), 0, MAX_PER_CLASS)
print(f"  {len(au_imgs)} görüntü yüklendi")

print("Tp (sahte) yükleniyor...")
tp_imgs, tp_labels = load_images(os.path.join(CASIA_PATH, "Tp"), 1, MAX_PER_CLASS)
print(f"  {len(tp_imgs)} görüntü yüklendi")

X = np.array(au_imgs + tp_imgs)
y = np.array(au_labels + tp_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# === CNN MODELİ ===
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# === LSTM MODELİ ===
def build_lstm():
    model = models.Sequential([
        layers.Input(shape=(128, 128*3)),  # Her satır = 128 piksel x 3 kanal
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# LSTM için veriyi reshape et
X_train_lstm = X_train.reshape(len(X_train), 128, 128*3)
X_test_lstm = X_test.reshape(len(X_test), 128, 128*3)

# === CNN EĞİTİMİ ===
print("\n=== CNN EĞİTİMİ ===")
cnn_model = build_cnn()
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

cnn_callbacks = [
    ModelCheckpoint('models/cnn_model.h5', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, restore_best_weights=True)
]

cnn_history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=cnn_callbacks
)

cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {cnn_acc:.4f}")

# === LSTM EĞİTİMİ ===
print("\n=== LSTM EĞİTİMİ ===")
lstm_model = build_lstm()
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.summary()

lstm_callbacks = [
    ModelCheckpoint('models/lstm_model.h5', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, restore_best_weights=True)
]

lstm_history = lstm_model.fit(
    X_train_lstm, y_train,
    validation_data=(X_test_lstm, y_test),
    epochs=20,
    batch_size=32,
    callbacks=lstm_callbacks
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_test_lstm, y_test)
print(f"LSTM Test Accuracy: {lstm_acc:.4f}")

print("\n✅ Eğitim tamamlandı!")
print(f"CNN modeli: models/cnn_model.h5")
print(f"LSTM modeli: models/lstm_model.h5")