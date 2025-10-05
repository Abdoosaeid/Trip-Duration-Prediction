import pandas as pd
import numpy as np
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


def euclidean_distance_km(df,lat1_col='pickup_latitude',lon1_col='pickup_longitude',lat2_col='dropoff_latitude',lon2_col='dropoff_longitude'):
    """
    Compute approximate Euclidean distance (in km) between pickup and dropoff
    points for all rows in a DataFrame (vectorized, no apply).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing pickup and dropoff coordinates.
    lat1_col, lon1_col : str
        Column names for pickup latitude and longitude.
    lat2_col, lon2_col : str
        Column names for dropoff latitude and longitude.

    Returns
    -------
    pandas.Series
        Distances in kilometers for each row.
    """
    km_per_degree_lat = 111  # km per degree latitude

    avg_lat_rad = np.radians((df[lat1_col] + df[lat2_col]) / 2)
    km_per_degree_lon = 111 * np.cos(avg_lat_rad)

    dx = (df[lon2_col] - df[lon1_col]) * km_per_degree_lon
    dy = (df[lat2_col] - df[lat1_col]) * km_per_degree_lat

    return np.sqrt(dx**2 + dy**2)

def add_log_transformed_feature(df, col="trip_duration", new_col=None):
    """
    Add a log1p-transformed version of a column to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    col : str, default="trip_duration"
        Column to transform.
    new_col : str or None, default=None
        Name of the new column. If None, will be "{col}_log1p".

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with new transformed column added.
    """
    if new_col is None:
        new_col = f"{col}_log1p"

    df[new_col] = np.log1p(df[col].values)
    return df
