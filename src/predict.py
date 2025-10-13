import pandas as pd
from utils import prepare_data,column_transformation
from sklearn.metrics import root_mean_squared_error, r2_score


def predict_eval(model, df, train_features,target, name):
    y_true = df[target]
    y_pred = model.predict(df[train_features])

    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"Model": name, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    import os, joblib

    path = r"D:\Trip-Duration-Prediction-\input\val.csv"
    df, new_column = prepare_data(path, target_column='trip_duration',remove_outlier=False)

    y_train = df[new_column]

    X_train = df.drop(columns=[new_column])
    column_transformer,train_features =   column_transformation(X_train)

    column_transformer.fit(X_train)

    models_dir = r"D:\Trip-Duration-Prediction\models"

    results = []

    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pkl"):
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)

            res = predict_eval(model, df,train_features,new_column, model_file)
            results.append(res)

    results_df = pd.DataFrame(results)
    print(results_df)