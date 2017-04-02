import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from clean import Clean
from statistic import Statistic
from plot import Plot

# Main
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

train, test = Clean(train, test).remove_constant()
train, test = Clean(train, test).remove_duplicates()

print train.head(),'\n'
print test.head(),'\n'

#print train.describe(),'\n'
#print test.describe(),'\n'

df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
print df

plt.plot(train[:100])
plt.ylabel('train')
plt.show()