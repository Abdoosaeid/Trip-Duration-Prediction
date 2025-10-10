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


def euclidean_distance_km(df,lat1_col='pickup_latitude',lon1_col='pickup_longitude',lat2_col='dropoff_latitude',lon2_col='dropoff_longitude',new_col='euclidean_distance'):
    """
      Compute approximate Euclidean distance (in km) between pickup and dropoff
      points and add it as a new column in the DataFrame.

      Parameters
      ----------
      df : pandas.DataFrame
          Input DataFrame containing pickup and dropoff coordinates.
      lat1_col, lon1_col : str
          Column names for pickup latitude and longitude.
      lat2_col, lon2_col : str
          Column names for dropoff latitude and longitude.
      new_col : str, default='euclidean_distance'
          Name of the new column to store computed distances.

      Returns
      -------
      pandas.DataFrame
          Same DataFrame with an additional column for Euclidean distance.
      """
    lat1 = df[lat1_col].to_numpy()
    lon1 = df[lon1_col].to_numpy()
    lat2 = df[lat2_col].to_numpy()
    lon2 = df[lon2_col].to_numpy()

    km_per_degree_lat = 111.0
    avg_lat_rad = np.radians((lat1 + lat2) * 0.5)
    km_per_degree_lon = km_per_degree_lat * np.cos(avg_lat_rad)

    dx = (lon2 - lon1) * km_per_degree_lon
    dy = (lat2 - lat1) * km_per_degree_lat

    df[new_col] = np.sqrt(dx * dx + dy * dy)

    return df


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
    df = df.drop(columns=[col])
    return df, new_col



def add_manhattan_distance(df):
    lat_dist = (df['pickup_latitude'] - df['dropoff_latitude']).abs() * 111
    lon_dist = (df['pickup_longitude'] - df['dropoff_longitude']).abs() * 111 * np.cos(np.radians(df['pickup_latitude']))
    df['manhattan_distance'] = lat_dist + lon_dist
    return df

def add_direction(df):
    delta_lon = np.radians(df['dropoff_longitude'] - df['pickup_longitude'])
    pickup_lat = np.radians(df['pickup_latitude'])
    drop_lat = np.radians(df['dropoff_latitude'])

    y = np.sin(delta_lon) * np.cos(drop_lat)
    x = (np.cos(pickup_lat) * np.sin(drop_lat) -
         np.sin(pickup_lat) * np.cos(drop_lat) * np.cos(delta_lon))

    bearing = np.degrees(np.arctan2(y, x))
    df['direction'] = (bearing + 360) % 360
    return df
