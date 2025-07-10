import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_game_data(path):
    data = np.load(path)
    boards = data['boards'] 
    labels = data['labels'] 
    boards = boards[..., np.newaxis]
    return boards, labels


def create_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for win probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, data, labels, epochs=100, batch_size=64):

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        data, labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop]
    )

def main():
    data, labels = load_game_data('data/othello_data.npz')  
    input_shape = (8, 8, 1) 
    model = create_model(input_shape)
    train_model(model, data, labels)
    model.save('othello_model.h5')

if __name__ == "__main__":
    main()