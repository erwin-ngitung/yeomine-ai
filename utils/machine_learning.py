import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json

PATH = '.'


def random_forest_model(file, data_predict, path_json=f'{PATH}/data/dataset/label_json.json'):
    dataset = pd.read_csv(file)
    data_json = json.load(open(path_json, 'rb'))

    df = dataset.copy()

    for col in data_json.keys():
        df[col] = df[col].replace(data_json[col])

    x = df[[col for col in df.columns if col != 'production']].values
    y = df['production'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44)

    rf_model = RandomForestRegressor(n_estimators=50, max_features="sqrt", random_state=44)
    rf_model.fit(x_train, y_train)

    y_predict = rf_model.predict(np.array(data_predict).reshape(1, -1))

    return y_predict

