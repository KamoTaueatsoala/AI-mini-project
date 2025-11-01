import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import copy

# ----- Model Definition -----
def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(224, 224, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.00018),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----- Smoothed Adaptive Confusion Callback -----
class AdaptiveConfusionCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, base_weights, max_adjust=0.02, alpha=0.2):
        """
        max_adjust: max fractional change per epoch (smaller = smoother updates)
        alpha: smoothing factor for exponential moving average of class weights
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.class_weights = base_weights.copy()
        self.max_adjust = max_adjust
        self.alpha = alpha

        # Track best epoch
        self.best_epoch = -1
        self.best_score = -1
        self.best_ratios = (0, 0)
        self.best_model_weights = None  # store best epoch weights

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.y_val, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        print(f"\nConfusion Matrix at Epoch {epoch+1}:\n{cm}\n")

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Per-class correct ratios
        normal_ratio = tn / max(1, tn + fp)
        abnormal_ratio = tp / max(1, tp + fn)
        print(f"Normal correct ratio: {normal_ratio:.3f}, Abnormal correct ratio: {abnormal_ratio:.3f}")

        # Best epoch selection: geometric mean
        score = normal_ratio * abnormal_ratio
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.best_ratios = (normal_ratio, abnormal_ratio)
            self.best_model_weights = self.model.get_weights()  # save best weights

        # Adaptive class weight adjustment
        if fn > fp:
            raw_factor = 1 + min((fn - fp) / max(1, fn + fp) * self.max_adjust, self.max_adjust)
            new_weight_abnormal = self.class_weights[1] * raw_factor
            new_weight_normal = self.class_weights[0] / raw_factor
        else:
            raw_factor = 1 + min((fp - fn) / max(1, fn + fp) * self.max_adjust, self.max_adjust)
            new_weight_normal = self.class_weights[0] * raw_factor
            new_weight_abnormal = self.class_weights[1] / raw_factor

        # Smoothed update
        self.class_weights[0] = (1 - self.alpha) * self.class_weights[0] + self.alpha * new_weight_normal
        self.class_weights[1] = (1 - self.alpha) * self.class_weights[1] + self.alpha * new_weight_abnormal

        # Normalize to sum=1
        total = self.class_weights[0] + self.class_weights[1]
        self.class_weights[0] /= total
        self.class_weights[1] /= total

        print(f"→ Smoothed class weights for next epoch: {self.class_weights}\n")

# ----- Training Function -----
def train_model(model, X_train, X_val, y_train, y_val, epochs=50):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Oversample abnormals
    abnormal_idx = np.where(y_train.argmax(axis=1) == 1)[0]
    oversample_factor = 1.0
    n_dup = int(len(abnormal_idx) * oversample_factor)
    dup_idx = np.random.choice(abnormal_idx, n_dup, replace=True)
    X_train_os = np.concatenate([X_train, X_train[dup_idx]], axis=0)
    y_train_os = np.concatenate([y_train, y_train[dup_idx]], axis=0)
    perm = np.random.permutation(len(X_train_os))
    X_train_os, y_train_os = X_train_os[perm], y_train_os[perm]

    # Initial class weights
    num_normal = np.sum(y_train_os.argmax(axis=1) == 0)
    num_abnormal = np.sum(y_train_os.argmax(axis=1) == 1)
    total = num_normal + num_abnormal
    base_weights = {0: 0.5427, 1: 0.4573}
    print("Initial class weights:", base_weights)

    # Adaptive callback
    adaptive_cb = AdaptiveConfusionCallback(X_val, y_val, base_weights)

    # Training loop
    history_accumulator = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        history = model.fit(
            datagen.flow(X_train_os, y_train_os, batch_size=16, shuffle=True),
            steps_per_epoch=len(X_train_os) // 16,
            epochs=1,
            validation_data=(X_val, y_val),
            class_weight=adaptive_cb.class_weights,
            verbose=1,
            callbacks=[adaptive_cb]
        )

        # Accumulate history
        for key in history_accumulator:
            if key in history.history:
                history_accumulator[key].extend(history.history[key])

    # Restore best model weights
    if adaptive_cb.best_model_weights is not None:
        model.set_weights(adaptive_cb.best_model_weights)
        print(f"\n✅ Restored model from best epoch {adaptive_cb.best_epoch} "
              f"with Normal ratio {adaptive_cb.best_ratios[0]:.3f} "
              f"and Abnormal ratio {adaptive_cb.best_ratios[1]:.3f}")

    # Return history in a simple object compatible with main.py
    class SimpleHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    return model, SimpleHistory(history_accumulator)
