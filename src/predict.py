import pandas as pd
from utils import prepare_data, column_transformation
from sklearn.metrics import r2_score
import os
import joblib

def predict_eval(model, df, train_features, target):
    """Evaluate model and return R² score."""
    y_true = df[target]
    y_pred = model.predict(df[train_features])
    return r2_score(y_true, y_pred)

if __name__ == "__main__":
    # === Paths ===
    base_path = r"D:\Trip-Duration-Prediction-\input"
    models_dir = r"D:\Trip-Duration-Prediction\models"

    # === File names ===
    files = {
        "train": os.path.join(base_path, "train.csv"),
        "val": os.path.join(base_path, "val.csv"),
    }

    # Dictionary to store results: {model_name: {"train": r2, "val": r2}}
    results_dict = {}

    # === Loop over datasets ===
    for dataset_name, path in files.items():
        print(f"\n--- Processing {dataset_name.upper()} dataset ---")

        # Prepare data
        df, target_col = prepare_data(path, target_column='trip_duration', remove_outlier=True)
        X_data = df.drop(columns=[target_col])

        # Column transformation
        column_transformer, train_features = column_transformation(X_data)
        column_transformer.fit(X_data)

        # Evaluate all models
        for model_file in os.listdir(models_dir):
            if model_file.endswith(".pkl"):
                model_path = os.path.join(models_dir, model_file)
                model = joblib.load(model_path)

                r2 = predict_eval(model, df, train_features, target_col)

                # Add result to dictionary
                if model_file not in results_dict:
                    results_dict[model_file] = {}
                results_dict[model_file][dataset_name] = r2

    # === Convert results to DataFrame ===
    results_df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    results_df.rename(columns={'index': 'Model'}, inplace=True)
    results_df = results_df[['Model', 'train', 'val']]  # Ensure consistent column order

    # === Display and save ===
    print("\n=== Final R² Results ===")
    print(results_df)

    save_path = os.path.join(r"D:\Trip-Duration-Prediction\reports", "r2_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nSaved R² results to: {save_path}")
