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

    #Remove outliers for a specif column in a dataframe
    #It is necessary specific the dataframe(0 for train and 1 > for test),
    #the column (must be a relevant column)
    #and the interval for identify the outliers (default is 1.5)
    def remove_outliers(self, df_select, column, interv=1.5):
        self.df_select = df_select #select the dataframe
        self.column = column #select the column to do the boxplot
        self.interv = interv #select the value to define the outliers interval
        
        #Do the boxplot for train data and remove outliers by column specified
        if(self.df_select == 0):            
            data = self.train.iloc[:,self.column].sort_values(ascending=True)
            data = data.values
            r = plt.boxplot(data, 0, 'gD', 1, self.interv)
            bottom_points = r["whiskers"][0].get_data()[1]
            top_points = r["whiskers"][1].get_data()[1]
            bottom_points = bottom_points[0]
            top_points = top_points[0]
            train_ = self.train[self.train.iloc[:,self.column] >= bottom_points]
            train_ = train_[train_.iloc[:,self.column] <= top_points]
            return train_
        else:#Do the boxplot for teste data
            data = self.test.iloc[:,self.column].sort_values(ascending=True)
            data = data.values
            r = plt.boxplot(data, 0, 'gD', 1, self.interv)
            bottom_points = r["whiskers"][0].get_data()[1]
            top_points = r["whiskers"][1].get_data()[1]
            bottom_points = bottom_points[0]
            top_points = top_points[0]
            test_ = self.test[self.test.iloc[:,self.column] >= bottom_points]
            test_ = test_[test_.iloc[:,self.column] <= top_points]
            return self.test
        return self.train, self.test

    