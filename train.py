import os
import numpy as np
import argparse
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def parse_args():
    parser = argparse.ArgumentParser(description='Entrena una red de valor para Otelo.')
    parser.add_argument('--data', default='data/othello_data.npz')
    parser.add_argument('--output', default='othello_model.keras')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--validation-split', type=float, default=0.15)
    return parser.parse_args()


def load_game_data(path):
    data = np.load(path)
    boards = data['boards'].astype(np.float32)
    labels = data['labels'].astype(np.float32)

    if boards.ndim == 3:
        boards = boards[..., np.newaxis]

    if boards.shape[-1] == 1:
        legacy_plane = boards[..., 0]
        black_plane = (legacy_plane == 1).astype(np.float32)
        white_plane = (legacy_plane == -1).astype(np.float32)
        turn_plane = np.zeros_like(black_plane, dtype=np.float32)
        boards = np.stack((black_plane, white_plane, turn_plane), axis=-1)

    return boards, labels


def create_model(input_shape, learning_rate=3e-4):
    regularizer = regularizers.l2(1e-4)

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='valid', kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.35),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.2),
        layers.Dense(1, activation='tanh'),
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(delta=0.5),
        metrics=['mae'],
    )
    return model


def train_model(model, data, labels, epochs=60, batch_size=256, validation_split=0.15):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
    ]

    return model.fit(
        data,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
    )


def main():
    args = parse_args()
    data, labels = load_game_data(args.data)
    input_shape = data.shape[1:]
    model = create_model(input_shape, learning_rate=args.learning_rate)
    train_model(
        model,
        data,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    model.save(args.output)


if __name__ == '__main__':
    main()