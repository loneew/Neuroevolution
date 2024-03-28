from keras_tuner.src.backend import keras
import keras_tuner as kt
import tensorflow as tf


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(10))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def keras_tuner(x_tr, y_tr, x_tst, y_tst, num_generations):
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='folder')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(x_tr, y_tr, epochs=num_generations, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"\tПошук гіперпараметрів завершено.\n- the optimal number of units in the first densely-connected layer :{best_hps.get('units')}\n- the optimal learning rate for the optimizer: {best_hps.get('learning_rate')}.")

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_tr, y_tr, epochs=num_generations, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(x_tr, y_tr, epochs=best_epoch, validation_split=0.2)

    eval_result = hypermodel.evaluate(x_tst, y_tst)
    print("[test loss, test accuracy]:", eval_result)
