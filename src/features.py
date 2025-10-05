import pandas as pd

def add_time_features(df, datetime_col="pickup_datetime", drop_original=True, extra_features=True):
    """
    Extract time-based features from a datetime column:
    - month
    - day
    - hour
    - dayofyear
    - dayofweek
    - optional: is_weekend, week, quarter

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    datetime_col : str, default="pickup_datetime"
        Column containing datetime values.
    drop_original : bool, default=True
        Whether to drop the original datetime column.
    extra_features : bool, default=True
        Whether to include extra features (is_weekend, week, quarter).

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with new features added.
    new_features : list
        List of names of the new features created.
    """

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # basic features
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["hour"] = df[datetime_col].dt.hour
    df["dayofyear"] = df[datetime_col].dt.dayofyear
    df["dayofweek"] = df[datetime_col].dt.dayofweek

    new_features = ["month", "day", "hour", "dayofyear", "dayofweek"]

    # optional extras
    if extra_features:
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["week"] = df[datetime_col].dt.isocalendar().week.astype(int)
        df["quarter"] = df[datetime_col].dt.quarter
        new_features += ["is_weekend", "week", "quarter"]

    if drop_original:
        df = df.drop(columns=[datetime_col])

    return df, new_features


