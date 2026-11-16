import pickle

from flask import Flask
from flask import request
from flask import jsonify

import xgboost as xgb



model_file = 'xGboost_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])

def predict():
    vehicle = request.get_json()

    X = dv.transform([vehicle])

    features = list(dv.get_feature_names_out())
    dtest = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dtest)
    failure= y_pred>0.015


    result = {
        'failure_probability': float(y_pred[0]),
        'failure': bool(failure)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)