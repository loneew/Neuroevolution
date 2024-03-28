import pandas as pd
import matplotlib.pyplot as plt


def graph_average_fitness(df):
    # Фільтруємо дані для NEAT та WANN
    neat_data = df[df['Algorithm'] == 'NEAT']
    wann_data = df[df['Algorithm'] == 'WANN']

    # Побудова графіка
    plt.plot(neat_data['Generation'], neat_data["Population's average fitness"], label='NEAT')
    plt.plot(wann_data['Generation'], wann_data["Population's average fitness"], label='WANN')

    # Додавання підписів до осей та заголовка
    plt.xlabel('Generation')
    plt.ylabel("Population's average fitness")
    plt.title("Population's average fitness Comparison")
    plt.legend()

    # Відображення графіка
    plt.show()


def graph_best_fitness(df):
    neat_data = df[df['Algorithm'] == 'NEAT']
    wann_data = df[df['Algorithm'] == 'WANN']

    plt.plot(neat_data['Generation'], neat_data["Best fitness"], label='NEAT')
    plt.plot(wann_data['Generation'], wann_data["Best fitness"], label='WANN')

    plt.xlabel('Generation')
    plt.ylabel("Best fitness")
    plt.title("Best fitness Comparison")
    plt.legend()

    plt.show()


def graph_adjusted_fitness(df):
    neat_data = df[df['Algorithm'] == 'NEAT']
    wann_data = df[df['Algorithm'] == 'WANN']

    plt.plot(neat_data['Generation'], neat_data["Average adjusted fitness"], label='NEAT')
    plt.plot(wann_data['Generation'], wann_data["Average adjusted fitness"], label='WANN')

    plt.xlabel('Generation')
    plt.ylabel("Average adjusted fitness")
    plt.title("Average adjusted fitness Comparison")
    plt.legend()

    plt.show()


def graph_stdev(df):
    neat_data = df[df['Algorithm'] == 'NEAT']
    wann_data = df[df['Algorithm'] == 'WANN']

    plt.plot(neat_data['Generation'], neat_data["stdev"], label='NEAT')
    plt.plot(wann_data['Generation'], wann_data["stdev"], label='WANN')

    plt.xlabel('Generation')
    plt.ylabel("stdev")
    plt.title("Stdev Comparison")
    plt.legend()

    plt.show()


def graph_genetic_distance(df):
    neat_data = df[df['Algorithm'] == 'NEAT']
    wann_data = df[df['Algorithm'] == 'WANN']

    plt.plot(neat_data['Generation'], neat_data["Mean genetic distance"], label='NEAT')
    plt.plot(wann_data['Generation'], wann_data["Mean genetic distance"], label='WANN')

    plt.xlabel('Generation')
    plt.ylabel("Mean genetic distance")
    plt.title("Mean genetic distance Comparison")
    plt.legend()

    plt.show()


def calulate_time(df):
    grouped_data = df.groupby('Algorithm')

    # Обчислюємо середній час для NEAT та WANN
    neat_average_time = grouped_data.get_group('NEAT')["Generation time, sec"].mean()
    wann_average_time = grouped_data.get_group('WANN')["Generation time, sec"].mean()

    print("Середній час NEAT:", neat_average_time, "секунд")
    print("Середній час WANN:", wann_average_time, "секунд")


def graph_accuracy(df):
    keras_data = df[df['Algorithm'] == 'Keras Tuner']

    plt.plot(keras_data['Epoch'], keras_data["accuracy"], label='accuracy')
    plt.plot(keras_data['Epoch'], keras_data["val_accuracy"], label='val_accuracy')

    plt.xlabel('Epoch')
    plt.title("Accuracy")
    plt.legend()

    plt.show()


def graph_loss(df):
    keras_data = df[df['Algorithm'] == 'Keras Tuner']

    plt.plot(keras_data['Epoch'], keras_data["loss"], label='loss')
    plt.plot(keras_data['Epoch'], keras_data["val_loss"], label='val_loss')

    plt.xlabel('Epoch')
    plt.title("Loss")
    plt.legend()

    plt.show()


df_1 = pd.read_excel('results.xlsx', sheet_name='neat_wann')

graph_average_fitness(df_1)
graph_best_fitness(df_1)
graph_adjusted_fitness(df_1)
graph_stdev(df_1)
graph_genetic_distance(df_1)
calulate_time(df_1)

df_2 = pd.read_excel('results.xlsx', sheet_name='keras')
graph_accuracy(df_2)
graph_loss(df_2)
