import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

output_file = 'xGboost_model.bin'

cost1=10
cost2=500
def cost(y_val,y_pred,t=0.015):
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    predict_positive = (y_pred >=t)
    predict_negative = (y_pred < t)

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    return cost1*fp+cost2*fn

df = pd.read_csv('data/aps_failure_training_set.csv')

df=df.rename(columns={'class': 'failure'})
df.failure = (df.failure == 'pos').astype(int)
df.failure.value_counts()
column_names=df.columns
column_names=column_names.drop('failure')
for col in column_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')

corr_val=df[column_names].corrwith(df.failure).abs()
feature = pd.DataFrame({ "corr_val": corr_val,"mean": df[column_names].mean()})
df=df.fillna(df[column_names].mean())
important_feature=feature[feature.corr_val>0.2].index.tolist()
important_feature.append('failure')
df_full_train=df[important_feature]

y_full_train = df_full_train.failure.values
del df_full_train['failure']

df_test = pd.read_csv('data/aps_failure_test_set.csv')
df_test=df_test.rename(columns={'class': 'failure'})
df_test.failure = (df_test.failure == 'pos').astype(int)
for col in column_names:
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
df_test=df_test.fillna(df[column_names].mean())
df_test=df_test[important_feature]
y_test = df_test.failure.values
del df_test['failure']

def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)

    dicts_train = df_train.to_dict(orient='records')
    X_train = dv.fit_transform(dicts_train)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    xgb_params = {
    'eta': 0.05, 
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=195,
                  verbose_eval=0)

    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)

    features = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dtest)

    return y_pred

dv, model =train(df_full_train, y_full_train)

y_pred=predict(df_test, dv, model)
cost=cost(y_test,y_pred)

print( f'cost on the test dataset is {cost}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')