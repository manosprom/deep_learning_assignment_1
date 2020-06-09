import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras_tqdm import TQDMNotebookCallback

from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D, BatchNormalization

np.random.seed(19870127)
tf.random.set_seed(19870127)

tqdmCallback = TQDMNotebookCallback(
    leave_inner=False,
    leave_outer=False,
)

tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=False,
                          write_images=False)

earlyStopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1
)

reduceLROnPlateau = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1, cooldown=2, min_delta=0.00001, min_lr=0
)

defaultCallbacks = [earlyStopping, reduceLROnPlateau]


def create_model(name=None, conv_layers=None, dense_layers=None, optimizer=None, print_summary=True):
    if not optimizer:
        optimizer = tf.keras.optimizers.Adam()

    model = Sequential(name=name)
    model.add(Input(shape=(28, 28, 1), name='icons_input'))

    if conv_layers is not None:
        for index, conv_layer in enumerate(conv_layers):
            filters = conv_layer.get("filters")
            kernel_size = conv_layer.get("kernel_size", (3, 3))
            strides = conv_layer.get("strides", (1, 1))
            padding = conv_layer.get("padding", 'same')
            dilation_rate = conv_layer.get("dilation_rate", (1, 1))
            activation = conv_layer.get("activation", "relu")
            conv_layer_name = "_".join(["conv", str(index), str(filters), ".".join(map(str, kernel_size)),
                                        ".".join(map(str, strides)), str(padding), ".".join(map(str, dilation_rate)), activation])

            model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation,
                             strides=strides, padding=padding, dilation_rate=dilation_rate, name=conv_layer_name))

            if conv_layer.get("batch_norm", False):
                model.add(BatchNormalization(name="_".join(["batch_norm", str(index)])))

            if conv_layer.get("pool", {}) is not None:
                pool = conv_layer.get("pool", {})
                pool_size = pool.get("size", (2,2))
                strides = pool.get("strides", pool_size)
                padding = pool.get("padding", "same")

                max_pool_layer_name = "_".join(["max_pool", str(index), ".".join(map(str, pool_size)),
                                                ".".join(map(str, strides)), str(padding), activation])

                model.add(MaxPool2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    name=max_pool_layer_name
                ))

            if conv_layer.get("dropout"):
                dropout_layer_name = "_".join(["dropout", str(index), str(conv_layer.get("dropout"))])
                model.add(Dropout(conv_layer.get("dropout"), name=dropout_layer_name))

    model.add(Flatten(name="cnn_flatten_connect"))
    # {"units": 256, "activation": "relu", "dropout": 0.2}
    if dense_layers is not None:
        for index, dense_layer in enumerate(dense_layers):
            units = dense_layer.get("units")
            activation = dense_layer.get("activation", "relu")

            dense_layer_name = "_".join(["dense", str(index), str(units), activation])
            model.add(Dense(units=units, activation=activation, name=dense_layer_name))

            if dense_layer.get("dropout"):
                dropout_layer_name = "_".join(["dropout", str(index), str(dense_layer.get("dropout"))])
                model.add(Dropout(dense_layer.get("dropout"), name=dropout_layer_name))

    model.add(Dense(10, activation=tf.nn.softmax, name='dense_output_softmax'))

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )
    if print_summary:
        print(model.summary())
        print("=======================")
        print(f'Compile with optimizers: {optimizer}')
        print("=======================")

    return model


def fit(
        train_data,
        val_data,
        model,
        batch_size=128,
        callbacks=None,
        epochs=300,
        print_summary=True,
        verbose=0
):
    # print("callbacks: ", callbacks)
    if not callbacks:
        callbacks = defaultCallbacks
    if print_summary:
        print(f'Fitting for {epochs} epochs \nwith {batch_size} batch_size and \ncallbacks{callbacks}')
    print("=======================")
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        workers=4,
        callbacks=callbacks,
        batch_size=batch_size,
        verbose=verbose
    )
    return history


def train(train_data, val_data, name="a_model", conv_layers=None, dense_layers=None, optimizer=None, callbacks=None,
          epochs=100, batch_size=128, verbose=0, print_summary=True):
    model = create_model(
        name=name, dense_layers=dense_layers, conv_layers=conv_layers, print_summary=print_summary, optimizer=optimizer,
    )
    # print(callbacks)
    history = fit(
        train_data,
        val_data,
        model,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        print_summary=print_summary,
        verbose=verbose
    )

    return model, history


def evaluate(model, test_data, test_labels, verbose=0):
    return model.evaluate(test_data, test_labels, verbose=verbose)


def clean_up(model):
    K.clear_session()
    del model
    gc.collect()


def create_stats(statistics):
    pd_create = []
    for stats in statistics:
        values = {
            "model_name": stats[0],
            "train_loss": stats[1].history["loss"][-1],
            "validation_loss": stats[1].history["val_loss"][-1],
            "test_loss": stats[2][0],
            "train_categorical_accuracy": stats[1].history["sparse_categorical_accuracy"][-1],
            "validation_categorical_accuracy": stats[1].history["val_sparse_categorical_accuracy"][-1],
            "test_categorical_accuracy": stats[2][1],
        }
        pd_create.append(values)
    return pd.DataFrame(pd_create)


def summary(history, evaluation):
    print("=======================")
    print(f'Train loss {history.history["loss"][-1]}')
    print(f'Validation loss {history.history["val_loss"][-1]}')
    print(f'Test loss {evaluation[0]}')
    print("=======================")
    print(f'Train categorical accuracy {history.history["sparse_categorical_accuracy"][-1]}')
    print(f'Validation categorical accuracy {history.history["val_sparse_categorical_accuracy"][-1]}')
    print(f'Test categorical accuracy {evaluation[1]}')
    print("=======================")
