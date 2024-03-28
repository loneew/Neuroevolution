from keras_tuner.src.backend import keras
from neuro_evolution import NEATClassifier, WANNClassifier
from sklearn.model_selection import train_test_split
from krs_tun import *

num_samples = 800
num_generations = 10
size_pop = 150


def normalization(x_tr, x_tst):
    x_tr = x_tr.astype('float32') / 255.0
    x_tst = x_tst.astype('float32') / 255.0

    return x_tr, x_tst


def neat_sklearn(x_tr, y_tr, x_tst, y_tst):
    # Зменшення обсягу даних
    x_train_subset = x_tr[:num_samples]
    y_train_subset = y_tr[:num_samples]

    # Перетворення даних для використання у WANN
    x_train_flattened = x_train_subset.reshape(x_train_subset.shape[0], -1)
    x_test_flattened = x_tst.reshape(x_tst.shape[0], -1)

    # Розділення на тренувальний та тестовий набори
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_flattened, y_train_subset,
                                                                              test_size=0.2, random_state=42)

    # Налаштування параметрів NEATClassifier
    clf = NEATClassifier(number_of_generations=num_generations,
                         fitness_threshold=0.9,
                         pop_size=size_pop)

    clf.fit(x_train_split, y_train_split)


def wann_sklearn(x_tr, y_tr, x_tst, y_tst):
    x_train_subset = x_tr[:num_samples]
    y_train_subset = y_tr[:num_samples]

    x_train_flattened = x_train_subset.reshape(x_train_subset.shape[0], -1)
    x_test_flattened = x_tst.reshape(x_tst.shape[0], -1)

    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_flattened, y_train_subset,
                                                                              test_size=0.2, random_state=42)

    clf = WANNClassifier(
        single_shared_weights=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, -2.0, -1.0],
        number_of_generations=num_generations,
        pop_size=size_pop,
        fitness_threshold=0.9,
        activation_default='relu')

    clf.fit(x_train_split, y_train_split)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = normalization(x_train, x_test)

    print("\t---NEAT---")
    neat_sklearn(x_train, y_train, x_test, y_test)

    #print("\t---WANN---")
    #neat_sklearn(x_train, y_train, x_test, y_test)

    #print("\t----Keras Tun---")
    #keras_tuner(x_train, y_train, x_test, y_test, num_generations)
