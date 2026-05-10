import pandas as pd
import os
from src.data_loader import align_data
from src.cleaner import impute_missing_sensor_values, handle_outliers


DAILY_ROLLING_WINDOW = '3D'
AMMONIA_SURGE_WINDOW = '6h'
CALENDAR_LAG_DAYS = 1


def _add_time_window_delta(df, value_col, time_col, output_col, window):
    """
    Adds a value delta calculated over a real time-based rolling window.
    """
    df[output_col] = float('nan')

    for _, group in df.groupby('pond_id', sort=False):
        valid_group = group.dropna(subset=[time_col]).sort_values(time_col)
        if valid_group.empty:
            continue

        series = valid_group.set_index(time_col)[value_col]
        rolling_delta = series.rolling(window=window, min_periods=2).apply(
            lambda values: values[-1] - values[0],
            raw=True
        )
        df.loc[valid_group.index, output_col] = rolling_delta.to_numpy()

    return df


def _add_calendar_lag(df, value_col, output_col, lag_days=CALENDAR_LAG_DAYS):
    """
    Adds a lag feature by matching the exact calendar day instead of the
    previous observed row.
    """
    lagged = df[['pond_id', 'date', value_col]].dropna(subset=['date']).copy()
    lagged['date'] = lagged['date'] + pd.Timedelta(days=lag_days)
    lagged = lagged.drop_duplicates(subset=['pond_id', 'date'], keep='last')
    lagged = lagged.rename(columns={value_col: output_col})

    if output_col in df.columns:
        df = df.drop(columns=[output_col])

    return pd.merge(df, lagged, on=['pond_id', 'date'], how='left')


def _add_time_based_rolling_mean(
    df,
    value_col,
    output_col,
    window=DAILY_ROLLING_WINDOW
):
    """
    Adds a rolling mean over a real elapsed-time window for each pond.
    """
    df[output_col] = float('nan')

    for _, group in df.groupby('pond_id', sort=False):
        valid_group = group.dropna(subset=['date']).sort_values('date')
        if valid_group.empty:
            continue

        series = valid_group.set_index('date')[value_col]
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        df.loc[valid_group.index, output_col] = rolling_mean.to_numpy()

    return df


def daily_aggregation(sensor_df):
    """
    Performs daily aggregation (mean, std, max, min) for specified sensor metrics.

    Args:
        sensor_df (pd.DataFrame): Cleaned sensor data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with daily aggregated features.
    """
    # Ensure 'date' column is datetime for grouping
    sensor_df['date'] = pd.to_datetime(sensor_df['date'])

    # Define sensor columns for aggregation
    sensor_cols_for_agg = [
        'ph',
        'turbidity',
        'mq135',
        'mq137',
        'temp_deep',
        'temp_shallow',
        'hardness',
        'rgb_r',
        'rgb_g',
        'rgb_b',
        'vertical_temp_diff',
        'rgb_brightness',
        'mq137_6hr_change',
    ]

    # Filter to include only columns present in the DataFrame
    sensor_cols_for_agg = [col for col in sensor_cols_for_agg if col in sensor_df.columns]

    # Group by pond_id and date, then aggregate
    daily_agg_df = sensor_df.groupby(['pond_id', 'date'])[sensor_cols_for_agg].agg(
        ['mean', 'std', 'max', 'min']
    )
    daily_agg_df.columns = ['_'.join(col).strip() for col in daily_agg_df.columns.values]
    daily_agg_df = daily_agg_df.reset_index()

    # Calculate daily pH range after aggregation
    if 'ph_max' in daily_agg_df.columns and 'ph_min' in daily_agg_df.columns:
        daily_agg_df['ph_range'] = daily_agg_df['ph_max'] - daily_agg_df['ph_min']

    return daily_agg_df

def create_domain_specific_features(sensor_df):
    """
    Creates domain-specific features like Vertical Temp Diff and Ammonia Surge.

    Args:
        sensor_df (pd.DataFrame): Cleaned sensor data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added domain-specific features.
    """
    if 'temp_shallow' in sensor_df.columns and 'temp_deep' in sensor_df.columns:
        sensor_df['vertical_temp_diff'] = sensor_df['temp_shallow'] - sensor_df['temp_deep']
    
    # RGB Brightness: sum R + G + B as a proxy indicator for water transparency
    if 'rgb_r' in sensor_df.columns and 'rgb_g' in sensor_df.columns and 'rgb_b' in sensor_df.columns:
        sensor_df['rgb_brightness'] = sensor_df['rgb_r'] + sensor_df['rgb_g'] + sensor_df['rgb_b']

    # Ammonia Surge: rate of change in mq137 over a real 6-hour window.
    if 'mq137' in sensor_df.columns and 'created_at' in sensor_df.columns:
        sensor_df = sensor_df.sort_values(by=['pond_id', 'created_at'])
        sensor_df = _add_time_window_delta(
            sensor_df,
            value_col='mq137',
            time_col='created_at',
            output_col='mq137_6hr_change',
            window=AMMONIA_SURGE_WINDOW
        )
    
    return sensor_df

def create_lag_rolling_features(combined_df):
    """
    Creates lag and rolling features using calendar-aware time windows.

    Args:
        combined_df (pd.DataFrame): DataFrame combined with daily logs and daily aggregated sensor data.
                                    Assumed to contain 'pond_id' and 'date'.

    Returns:
        pd.DataFrame: DataFrame with added lag and rolling features.
    """
    # Ensure sorted for correct lag/rolling calculations
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values(by=['pond_id', 'date'])

    # Lag-1: exact previous calendar day's death_count and feeding_amount.
    if 'death_count' in combined_df.columns:
        combined_df = _add_calendar_lag(
            combined_df,
            value_col='death_count',
            output_col='death_count_lag1'
        )
    if 'feeding_amount' in combined_df.columns:
        combined_df = _add_calendar_lag(
            combined_df,
            value_col='feeding_amount',
            output_col='feeding_amount_lag1'
        )

    # Rolling-3: elapsed 3-day moving average of Ammonia and pH volatility.
    if 'mq137_mean' in combined_df.columns:
        combined_df = _add_time_based_rolling_mean(
            combined_df,
            value_col='mq137_mean',
            output_col='mq137_mean_roll3'
        )
    # Calculate the 3-day average pH fluctuation range (Max - Min)
    if 'ph_range' in combined_df.columns:
        combined_df = _add_time_based_rolling_mean(
            combined_df,
            value_col='ph_range',
            output_col='ph_range_roll3'
        )
    
    return combined_df.sort_values(by=['pond_id', 'date'])

def build_features_and_split(sensor_data_path, pond_daily_logs_path, output_dir='data/processed', test_split_ratio=0.2):
    """
    Builds features, merges with daily logs, and performs a chronological train/test split.

    Args:
        sensor_data_path (str): Path to the sensor data CSV file.
        pond_daily_logs_path (str): Path to the pond daily logs CSV file.
        output_dir (str): Directory to save processed CSVs.
        test_split_ratio (float): Ratio of data to use for the test set (e.g., 0.2 for 20%).

    Returns:
        tuple: (train_df, test_df)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and Clean Data
    print("Step 1: Loading and cleaning data...")
    sensor_data_aligned, pond_daily_logs_df = align_data(sensor_data_path, pond_daily_logs_path)
    sensor_data_imputed = impute_missing_sensor_values(sensor_data_aligned.copy())
    sensor_data_cleaned = handle_outliers(sensor_data_imputed.copy())
    print("Data loading and cleaning complete.")

    #####################################################################################################################
    # Compute first, then aggregate to capture spikes, extreme temperature differences, and intense fluctuations 
    # (which are more suitable for predicting mortality rates).
    
    # 2. Create Domain-Specific Features
    print("\nStep 2: Creating domain-specific features on high-frequency data...")
    sensor_data_with_domain_features = create_domain_specific_features(sensor_data_cleaned.copy())
    print("Domain-specific features created.")

    # 3. Perform Daily Aggregation
    print("\nStep 3: Performing daily aggregation...")
    daily_features_df = daily_aggregation(sensor_data_with_domain_features.copy())
    print("Daily aggregation complete.")
    #####################################################################################################################

    # 4. Merge Daily Features with Pond Daily Logs
    print("\nStep 4: Merging daily features with pond daily logs...")
    # Ensure 'date' column in pond_daily_logs_df is also datetime
    pond_daily_logs_df['date'] = pd.to_datetime(pond_daily_logs_df['log_date']).dt.date
    pond_daily_logs_df['date'] = pd.to_datetime(pond_daily_logs_df['date']) # Convert to datetime objects for consistent merging

    # Convert medication_given to numeric (True = 1, False = 0)
    if 'medication_given' in pond_daily_logs_df.columns:
        pond_daily_logs_df['medication_given'] = pond_daily_logs_df['medication_given'].astype(int)

    merged_daily_features = pd.merge(daily_features_df, pond_daily_logs_df[['pond_id', 'date', 'death_count', 'feeding_amount', 'medication_given']], 
                                   on=['pond_id', 'date'], how='left')
    print("Merge complete.")

    # 5. Create Lag and Rolling Features
    print("\nStep 5: Creating lag and rolling features...")
    final_features_df = create_lag_rolling_features(merged_daily_features.copy())
    print("Lag and rolling features created.")

    # Persistence Step 1: Save merged_daily_features.csv
    merged_output_path = os.path.join(output_dir, 'merged_daily_features.csv')
    final_features_df.to_csv(merged_output_path, index=False)
    print(f"\nMerged daily features saved to: {merged_output_path}")

    # 6. Chronological Train/Test Split
    print("\nStep 6: Performing chronological train/test split...")
    final_features_df = final_features_df.sort_values(by='date').reset_index(drop=True)
    
    split_index = int(len(final_features_df) * (1 - test_split_ratio))
    train_df = final_features_df.iloc[:split_index]
    test_df = final_features_df.iloc[split_index:]

    print(f"Train set size: {len(train_df)} records, from {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test set size: {len(test_df)} records, from {test_df['date'].min()} to {test_df['date'].max()}")

    # Task F: Feature & Label Validation
    if 'death_count' not in test_df.columns:
        print("Warning: 'death_count' missing from test_df, but it is required for evaluation.")
    else:
        print("'death_count' present in test_df for evaluation.")


    # Persistence Step 2: Save train_set.csv and test_set.csv
    train_output_path = os.path.join(output_dir, 'train_set.csv')
    test_output_path = os.path.join(output_dir, 'test_set.csv')
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"Train set saved to: {train_output_path}")
    print(f"Test set saved to: {test_output_path}")

    return train_df, test_df

if __name__ == '__main__':
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'
    OUTPUT_DATA_DIR = 'data/processed'

    print("--- Starting Feature Building and Data Splitting ---")
    train_set, test_set = build_features_and_split(
        SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH, OUTPUT_DATA_DIR
    )
    print("\n--- Feature Building and Data Splitting Complete ---")

    print("\nTrain set head:")
    print(train_set.head())
    print("\nTest set head:")
    print(test_set.head())
