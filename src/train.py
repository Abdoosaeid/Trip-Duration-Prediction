from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import joblib
from utils import column_transformation, prepare_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
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
    print("✅ Model saved successfully at models/linear_regression.pkl")


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


    save_path = fr"D:\Trip-Duration-Prediction\models\polynomial_deg{degree}.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Polynomial Regression (degree={degree}) model saved successfully at {save_path}")

def lasso(data_path):

    df, new_column = prepare_data(data_path, target_column='trip_duration')


    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])

    column_transformer, _ = column_transformation(X_train)

    param_grid = {
        'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    lasso_cv = GridSearchCV(Lasso(),param_grid,cv=3,n_jobs=-1)
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('lasso', lasso_cv)
    ])


    model = pipeline.fit(X_train, y_train)


    joblib.dump(model, r"D:\Trip-Duration-Prediction\models\Lasso.pkl")
    print("✅ Model saved successfully at models/model_lasso.pkl")


def ridge(data_path):

    df, new_column = prepare_data(data_path, target_column='trip_duration')


    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])

    column_transformer, _ = column_transformation(X_train)

    param_grid = {
        'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    }
    lasso_cv = GridSearchCV(Ridge(),param_grid,cv=3,n_jobs=-1)
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('ridge', lasso_cv)
    ])


    model = pipeline.fit(X_train, y_train)


    joblib.dump(model, r"D:\Trip-Duration-Prediction\models\Ridge.pkl")
    print("✅ Model saved successfully at models/model_ridge.pkl")

def elastic_net(data_path):
    # Prepare data
    df, new_column = prepare_data(data_path, target_column='trip_duration')

    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])

    # Column transformation
    column_transformer, _ = column_transformation(X_train)

    # Correct parameter names
    param_grid = {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    }

    # Use ElasticNet with GridSearchCV
    elastic_cv = GridSearchCV(ElasticNet(), param_grid, cv=3, n_jobs=-1)

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('elastic', elastic_cv)
    ])

    # Train model
    model = pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(model, r"D:\Trip-Duration-Prediction\models\ElasticNet.pkl")
    print("✅ Model saved successfully at models/model_ElasticNet.pkl")


if __name__ == "__main__":
    data_path = r"D:\Trip-Duration-Prediction\data\raw\train.csv"
    Linear_regression(data_path)
    Polynomial_regression(data_path,6)
    lasso(data_path)
    ridge(data_path)
    elastic_net(data_path)
