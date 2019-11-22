import pandas as pd

from clean import Clean
from statistic import Statistic
from plot import Plot
from reduction import Reduction
from classifier import Classifier

# Main
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")



train, test = Clean(train, test).remove_constant()

#Plot(train, test).plot_line()

train, test = Clean(train, test).remove_duplicates()

#Plot(train, test).plot_line()

train, test, target, test_id = Reduction(train, test).principal_component_analysis()

Classifier(train, test, target, test_id).xgboost() #Com PCA
#Classifier(train, test, '', '').xgboost() #Sem PCA
