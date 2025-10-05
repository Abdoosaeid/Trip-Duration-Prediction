import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats


def load_data(dataset_path):

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"This file doesn't exist: {dataset_path}")

    df = pd.read_csv(dataset_path)

    return df

def split_data(df,target_column):

    X_train = df.drop(columns=target_column)
    y_train = df[target_column]

    return X_train, y_train

def column_transformation(df,numeric_features=None,categorical_features=None, scaler="standard", remainder="drop"):
    """
    Creates a ColumnTransformer for preprocessing numeric and categorical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    numeric_features : list, optional
        List of numeric feature names. If None, will auto-detect numeric columns.
    categorical_features : list, optional
        List of categorical feature names. If None, will auto-detect object/category columns.
    scaler : str or transformer, optional
        Which scaler to use for numeric features.
        Options: "standard", "minmax", "robust", or pass a custom scaler instance.
    remainder : str, optional
        How to handle other columns not specified ('drop' or 'passthrough').

    Returns
    -------
    column_transformer : ColumnTransformer
        Transformer with OHE for categoricals and scaling for numerics.
    train_features : list
        Combined list of features used for training.
    """

    # detect features if not provided
    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()

    train_features = categorical_features + numeric_features

    # choose scaler
    if isinstance(scaler, str):
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        if scaler not in scaler_map:
            raise ValueError(f"Scaler '{scaler}' not recognized. Use one of {list(scaler_map.keys())}.")
        scaler = scaler_map[scaler]

    # numeric pipeline
    num_pipeline = Pipeline(steps=[
         ('scaler', scaler)
    ])

    # categorical pipeline
    cat_pipeline = Pipeline(steps=[
         ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', num_pipeline, numeric_features),
            ('categorical', cat_pipeline, categorical_features)
        ],
        remainder=remainder
    )

    return column_transformer


def remove_outliers(df, feature_col=None, method='zscore', factor=1.5):
    """
     outlier removal function.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_col : str or None, optional
        Column to apply outlier removal on. If None, applies to all numeric columns.
    method : str, default='zscore'
        Method to detect outliers: 'zscore' or 'iqr'.
    factor : float, default=1.5
        Threshold factor (IQR multiplier or z-score cutoff).

    Returns
    -------
    df_cleaned : pd.DataFrame
        DataFrame with outliers removed.
    """

    df_clean = df.copy()


    if feature_col:
        cols = [feature_col]
    else:
        cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df_clean[col]))
            mask = z_scores < factor
        elif method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
        else:
            raise ValueError("Method must be 'zscore' or 'iqr'")

        df_clean = df_clean[mask]

    return df_clean