from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
from utils import column_transformation, prepare_data

def Linear_regression(data_path):

    df, new_column = prepare_data(data_path, target_column='trip_duration')

    print(
        df.info()
    )

    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])

    column_transformer, _ = column_transformation(X_train)


    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', LinearRegression())
    ])


    model = pipeline.fit(X_train, y_train)


    joblib.dump(model, r"D:\Trip-Duration-Prediction\models\model1_linear_regression.pkl")
    print("✅ Model saved successfully at models/model1_linear_regression.pkl")


if __name__ == "__main__":
    data_path = r"D:\Trip-Duration-Prediction\data\raw\train.csv"
    Linear_regression(data_path)
