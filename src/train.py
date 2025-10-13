from sklearn.linear_model import LinearRegression
import joblib
from utils import column_transformation, prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def Linear_regression(data_path):

    df, new_column = prepare_data(data_path, target_column='trip_duration')


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


def Polynomial_regression(data_path, degree=6):

    df, new_column = prepare_data(data_path, target_column='trip_duration')


    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])


    poly_features = ['euclidean_distance', 'manhattan_distance', 'direction','pickup_longitude']

    # Other numeric & categorical columns
    numeric_cols = [
         'passenger_count', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'month', 'day', 'hour',
        'dayofyear', 'dayofweek', 'is_weekend', 'week', 'quarter'
    ]
    categorical_cols = ['store_and_fwd_flag','vendor_id']


    preprocessor = ColumnTransformer(transformers=[
        ('poly', Pipeline([
            ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler())
        ]), poly_features),

        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])


    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regression', LinearRegression())
    ])


    model = pipeline.fit(X_train, y_train)


    save_path = fr"D:\Trip-Duration-Prediction\models\model_polynomial_deg{degree}.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Polynomial Regression (degree={degree}) model saved successfully at {save_path}")

if __name__ == "__main__":
    data_path = r"D:\Trip-Duration-Prediction\data\raw\train.csv"
    Linear_regression(data_path)
    Polynomial_regression(data_path,6)
