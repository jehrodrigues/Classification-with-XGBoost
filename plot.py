import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot Data
class Plot(object):

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def plot_line(self):
        print(self.train.head(),'\n')
        print(self.test.head(),'\n')

        #print train.describe(),'\n'
        #print test.describe(),'\n'

        df = pd.DataFrame(self.train.TARGET.value_counts())
        df['Percentage'] = 100*df['TARGET']/self.train.shape[0]
        print(df)

        plt.plot(self.train[:100])
        plt.ylabel('train')
        plt.show()

    def plot_histogram(self):
        #Plotar Histograma
        return 'histogram'

    def plot_pizza(self):
        #Plota Pizza
        return 'pizza'

    def plot_scatter(self):
        #Plotar Scater
        return 'scatter'