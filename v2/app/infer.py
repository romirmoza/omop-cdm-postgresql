import datetime
from operator import add
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import load

ROOT = "/"

class OmopParser(object):

    def __init__(self):
        self.name = 'omop_parser'
        self.X = 0
        self.y = 0
        self.person_id = 0
        
    def load_data(self, test_filename, train_filename):
        test = pd.read_csv(test_filename,low_memory = False)
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(test.race_concept_name)
        test.race_concept_name = label_encoder.transform(test.race_concept_name)
        test = test.fillna(0)
        self.person_id = X.person_id
        # order the columnns of the test set like the train set
        train_features = pd.read_csv(train_filename, nrows=1).columns.values
        X = X[train_features]
        X = test.drop(['death_in_next_window','person_id'], axis = 1)
        y = test[['death_in_next_window']]
        self.X = np.array(X)
        self.y = np.array(y).ravel()

        return

    def xgb_predict(self):
        '''infer with XGBoost'''
        import xgboost as xgb

        xgb_model =  load('/model/xgb_model.joblib')
        Y_pred = xgb_model.predict(self.X)
        output = pd.DataFrame(Y_pred,columns = ['score'])
        output_prob = pd.concat([self.person_id,output],axis = 1)
        output_prob.columns = ["person_id", "score"]
        output_prob.to_csv('/output/predictions.csv', index = False)
        print("Inferring stage finished", flush = True)

        return

if __name__ == '__main__':
    FOLDER ='scratch/'
    TEST_FILE_STR = 'test_all_nw.csv'
    TRAIN_FILE_STR = 'train_all_nw.csv'
    op = OmopParser()
    op.load_data(ROOT+FOLDER+TEST_FILE_STR, ROOT+FOLDER+TRAIN_FILE_STR)
    op.xgb_predict(ROOT+FOLDER+'infer.csv')
