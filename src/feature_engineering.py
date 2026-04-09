import pandas as pd
from data_loader import align_data
from cleaner import impute_missing_sensor_values, handle_outliers

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
    sensor_cols_for_agg = ['ph', 'turbidity', 'mq135', 'mq137', 'temp_deep', 'temp_shallow', 'hardness', 'rgb_r', 'rgb_g', 'rgb_b', 'vertical_temp_diff', 'rgb_brightness']

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

    # Ammonia Surge: Rate of change in mq137 over the last 6 hours.
    # This requires sorting by time within each pond and then applying a rolling difference.
    if 'mq137' in sensor_df.columns and 'created_at' in sensor_df.columns:
        sensor_df = sensor_df.sort_values(by=['pond_id', 'created_at'])
        # Calculate difference over a 6-hour window (approx 6*60/sampling_rate if sampling is per minute)
        # Assuming sensor_df is already at a relatively high frequency (e.g., minutes).
        # We'll use a 6-hour rolling window to calculate the change.
        # This is a simplification; a more robust solution would consider actual time differences.
        sensor_df['mq137_6hr_change'] = sensor_df.groupby('pond_id')['mq137'].diff(periods=6*60) # Assuming minutely data
    
    return sensor_df
def create_lag_rolling_features(combined_df):
    """
    Creates lag and rolling features.

    Args:
        combined_df (pd.DataFrame): DataFrame combined with daily logs and daily aggregated sensor data.
                                    Assumed to be sorted by 'pond_id' and 'date'.

    Returns:
        pd.DataFrame: DataFrame with added lag and rolling features.
    """
    # Ensure sorted for correct lag/rolling calculations
    combined_df = combined_df.sort_values(by=['pond_id', 'date'])

    # Lag-1: Previous day's death_count and feeding_amount
    if 'death_count' in combined_df.columns:
        combined_df['death_count_lag1'] = combined_df.groupby('pond_id')['death_count'].shift(1)
    if 'feeding_amount' in combined_df.columns:
        combined_df['feeding_amount_lag1'] = combined_df.groupby('pond_id')['feeding_amount'].shift(1)

    # Rolling-3: 3-day moving average of Ammonia (mq137_mean) and pH volatility (ph_std)
    # Assuming daily_agg_df is already merged and relevant columns exist
    if 'mq137_mean' in combined_df.columns:
        combined_df['mq137_mean_roll3'] = combined_df.groupby('pond_id')['mq137_mean'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    # Calculate the 3-day average pH fluctuation range (Max - Min)
    if 'ph_range' in combined_df.columns:
        combined_df['ph_range_roll3'] = combined_df.groupby('pond_id')['ph_range'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    
    return combined_df
if __name__ == '__main__':
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'

    print("Step 1: Aligning data...")
    sensor_data_aligned, pond_daily_logs_df = align_data(SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH)
    print("Alignment complete.")

    print("\nStep 2: Cleaning sensor data (imputation and outlier handling)...")
    sensor_data_imputed = impute_missing_sensor_values(sensor_data_aligned.copy())
    sensor_data_cleaned = handle_outliers(sensor_data_imputed.copy())
    print("Cleaning complete.")

    print("\nStep 3: Creating domain-specific features on high-frequency data...")
    sensor_data_with_domain_features = create_domain_specific_features(sensor_data_cleaned.copy())
    print("Domain-specific features created.")
    print("Sample with vertical_temp_diff and mq137_6hr_change:")
    print(sensor_data_with_domain_features[['pond_id', 'created_at', 'temp_shallow', 'temp_deep', 'vertical_temp_diff', 'mq137', 'mq137_6hr_change']].head())

    print("\nStep 4: Performing daily aggregation...")
    daily_features_df = daily_aggregation(sensor_data_with_domain_features.copy())
    print("Daily aggregation complete.")
    print("Daily aggregated features head:")
    print(daily_features_df.head())

    print("\nStep 5: Merging daily features with pond daily logs...")
    # Ensure 'date' column in pond_daily_logs_df is also datetime
    pond_daily_logs_df['date'] = pd.to_datetime(pond_daily_logs_df['log_date']).dt.date
    pond_daily_logs_df['date'] = pd.to_datetime(pond_daily_logs_df['date']) # Convert to datetime objects for consistent merging

    # Convert medication_given to numeric (True = 1, False = 0)
    if 'medication_given' in pond_daily_logs_df.columns:
        pond_daily_logs_df['medication_given'] = pond_daily_logs_df['medication_given'].astype(int)

    combined_df = pd.merge(daily_features_df, pond_daily_logs_df[['pond_id', 'date', 'death_count', 'feeding_amount', 'medication_given']], 
                           on=['pond_id', 'date'], how='left')
    print("Merge complete.")
    print("Combined DataFrame head after merge:")
    print(combined_df.head())

    print("\nStep 6: Creating lag and rolling features...")
    final_features_df = create_lag_rolling_features(combined_df.copy())
    print("Lag and rolling features created.")
    print("Final features DataFrame head:")
    print(final_features_df.head())

    print("\nFinal features DataFrame Info:")
    final_features_df.info()
    print("\nMissing values in final_features_df after all steps:")
    print(final_features_df.isnull().sum())
