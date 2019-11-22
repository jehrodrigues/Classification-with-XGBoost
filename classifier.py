import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

# Classifier XGBoost
class Classifier(object):

    def __init__(self, train, test, target, test_id):
        self.train = train
        self.test = test
        if(target == ''): #Teste sem PCA
            self.test_id = self.test.ID
            self.target = self.train.TARGET.values
        else: #Teste com PCA
            print('Teste com PCA')
            self.target = target
            self.test_id = test_id

    """
    XGBoost
    """
    def xgboost(self):
        X_train, X_test, y_train, y_test=train_test_split(self.train,self.target, test_size=0.2, random_state=0)
        scores = cross_val_score(xgb.XGBClassifier(), X_train, y_train, scoring='roc_auc', cv=3)
        print("Desempenho Curva ROC: {:.3f}\n".format(scores.mean()))

        #eval_metrics = ['auc']
        #eval_sets = [(X_train, y_train), (X_test, y_test)]

        xgb_m2=xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, min_child_weight=4, gamma=0, colsample_bytree=0.7, subsample=0.6, reg_alpha=5e-05, objective='binary:logistic', scale_pos_weight=1, seed=0)

        #xgb_m2.fit(X_train, y_train) #, eval_metric=eval_metrics, eval_set=eval_sets

        xgb_m2.fit(X_train, y_train, eval_metric="auc", verbose=False, eval_set=[(X_test, y_test)])

        # Criacao arquivo de predicoes
        y_pred=xgb_m2.predict_proba(self.test)
        print('y_pred: ', y_pred)

        # calculate the auc score
        print("Roc AUC: ", roc_auc_score(y_test, xgb_m2.predict_proba(X_test)[:, 1], average='macro'))

        df_submit=pd.DataFrame({'ID': self.test_id, 'TARGET': y_pred[:,1]})
        df_submit.to_csv('predicoes.csv', index=False)

        print(df_submit.head())