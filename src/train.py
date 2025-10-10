from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from utils import column_transformation, prepare_data

def Linear_regression(data_path):
    # إعداد البيانات
    df, new_column = prepare_data(data_path, target_column='trip_duration')

    # فصل الهدف (target)
    y_train = df[new_column]
    X_train = df.drop(columns=[new_column])

    # تجهيز الـ column transformer على الداتا بدون الهدف
    column_transformer, train_feature = column_transformation(X_train)

    # بناء الـ pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', column_transformer),
        ('regression', LinearRegression())
    ])

    # تدريب الموديل
    model = pipeline.fit(X_train, y_train)

    # التقييم على نفس الداتا
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)

    print("📊 Model Performance on Training Data:")
    print(f"   MSE  = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")

    # حفظ الموديل
    joblib.dump(model, r"D:\Trip-Duration-Prediction\models\model1_linear_regression.pkl")
    print("✅ Model saved successfully at models/model1_linear_regression.pkl")

    return model, mse, rmse, r2


if __name__ == "__main__":
    data_path = r"D:\Trip-Duration-Prediction\data\raw\train.csv"
    Linear_regression(data_path)
