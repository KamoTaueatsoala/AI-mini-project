import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(224, 224, 1)),  # Explicit input layer
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),  # Increased units
        Dropout(0.5),  # Increased dropout
        Dense(2, activation='softmax')  # Softmax for better probability distribution
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, X_test, y_train, y_test, datagen):
    # Calculate class weights
    class_weights = {0: 1.0, 1: 2.0}  # Higher weight for minority class (Abnormal)
    
    # Train with data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=16),  # Smaller batch size
                       steps_per_epoch=len(X_train) // 16,
                       epochs=50,  # More epochs
                       validation_data=(X_test, y_test),
                       class_weight=class_weights,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return model, history