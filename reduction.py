import itertools

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Dimensionality Reduction
class Reduction(object):

    def __init__(self, train, test):
        self.train = train
        self.test = test

    """
    A Analise de Componentes Principais (PCA) identifica a combinacao de atributos 
    (componentes principais ou direcoes no espaco de recursos) que representam a maior variacao nos dados. 
    Vamos calcular os 2 primeiros componentes principais dos dados de treinamento e, em seguida, 
    criar um grafico de dispersao visualizando os exemplos de dados de treinamento projetados nos 
    componentes calculados.
    """
    def principal_component_analysis(self):

        # Retirar atributos ID e TARGET
        test_id = self.test.ID
        x_test = self.test.drop(["ID"],axis=1)

        x_train = self.train.drop(["TARGET","ID"],axis=1)
        y_train = self.train["TARGET"]
        target = self.train.TARGET.values

        features = self.train.columns[1:-1]

        classes = np.sort(np.unique(y_train))
        labels = ["Clientes satisfeitos", "Clientes insatisfeitos"]

        # Normalize cada caracteristica a norma da unidade (comprimento do vetor)
        x_train_normalized = normalize(self.train[features], axis=0)
        x_test_normalized = normalize(x_test[features], axis=0)

        # Executar PCA
        pca = PCA(n_components=3)
        x_train_projected = pca.fit_transform(x_train_normalized)
        x_test_projected = pca.fit_transform(x_test_normalized)

        x_train.insert(1, 'PCAOne', x_train_projected[:, 0])
        x_train.insert(1, 'PCATwo', x_train_projected[:, 1])
        x_train.insert(1, 'PCAThree', x_train_projected[:, 2])

        x_test.insert(1, 'PCAOne', x_test_projected[:, 0])
        x_test.insert(1, 'PCATwo', x_test_projected[:, 1])
        x_test.insert(1, 'PCAThree', x_test_projected[:, 2])

        # Visualizar
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        colors = [(0.0, 0.63, 0.69), 'black']
        markers = ["o", "D"]
        for class_ix, marker, color, label in zip(
                classes, markers, colors, labels):
            ax.scatter(x_train_projected[np.where(y_train == class_ix), 0], #x_train_projected #x_train_normalized
                       x_train_projected[np.where(y_train == class_ix), 1], #x_train_projected #x_train_normalized
                       marker=marker, color=color, edgecolor='whitesmoke',
                       linewidth='1', alpha=0.9, label=label)
            ax.legend(loc='best')
        plt.title(
            "Diagrama de dispersao dos exemplos de dados de treinamento projetados nos 2 "
            "primeiros componentes principais")
        plt.xlabel("Eixo principal 1 - Explica %.1f %% da variancia" % (
            pca.explained_variance_ratio_[0] * 100.0))
        plt.ylabel("Eixo principal 2 - Explica %.1f %% da variancia" % (
            pca.explained_variance_ratio_[1] * 100.0))
        plt.show()

        plt.savefig("pca.pdf", format='pdf')
        plt.savefig("pca.png", format='png')
        return x_train, x_test, target, test_id