import numpy as np

# clean data
class Clean(object):

    def __init__(self, train, test):
        self.train = train
        self.test = test

    # Remove colunas com valores constantes (std = 0)
    # Para valores iguais na coluna, desvio padrao e 0
    def remove_constant(self):
        remove = []
        for col in self.train.columns:
            if self.train[col].std() == 0:
                remove.append(col)
        #axis: 1 para olhar coluna e 0 para linha
        #inplace: faz o replace na propria instancia
        self.train.drop(remove, axis=1, inplace=True)
        self.test.drop(remove, axis=1, inplace=True)
        return self.train, self.test

    # Remove colunas duplicadas
    def remove_duplicates(self):
        remove = []
        cols = self.train.columns
        for i in range(len(cols)-1):
            v = self.train[cols[i]].values
            for j in range(i+1,len(cols)):
                if np.array_equal(v,self.train[cols[j]].values):
                    remove.append(cols[j])

        self.train.drop(remove, axis=1, inplace=True)
        self.test.drop(remove, axis=1, inplace=True)
        return self.train, self.test

    def remove_absent(self):
        #Implementar tecnica para preencher o valor ausente
        return self.train, self.test
    
    def remove_inconsistent(self):
        #Implementar tecnica para resolver inconsistencia
        return self.train, self.test

    def remove_noise(self):
        #Implementar tecnica para tratar ruido
        return self.train, self.test

    def remove_outliers(self):
        #Implementar tecnica para remover outlier (plotar para descobrir os outliers)
        return self.train, self.test

    