import pandas as pd

def align_data(sensor_data_path, pond_daily_logs_path, min_records_per_day=10):
    """
    Loads sensor data and pond daily logs, aligns them by date and pond_id,
    handles time zones, extracts dates, and performs data integrity checks.

    Args:
        sensor_data_path (str): Path to the sensor data CSV file.
        pond_daily_logs_path (str): Path to the pond daily logs CSV file.
        min_records_per_day (int): Minimum number of sensor records required per day per pond.

    Returns:
        tuple: (aligned_sensor_data_df, pond_daily_logs_df)
            aligned_sensor_data_df (pd.DataFrame): Processed sensor data.
            pond_daily_logs_df (pd.DataFrame): Processed pond daily logs.
    """
    # 1.1 Load Data
    sensor_data = pd.read_csv(sensor_data_path)
    pond_daily_logs = pd.read_csv(pond_daily_logs_path)

    # Convert to datetime
    sensor_data['created_at'] = pd.to_datetime(sensor_data['created_at'])
    pond_daily_logs['log_date'] = pd.to_datetime(pond_daily_logs['log_date'])

    # 1.2 Time Zone Handling & Date Extraction
    # Convert to Taiwan time zone (+0800)
    # Ensure 'created_at' is timezone-aware before converting
    if sensor_data['created_at'].dt.tz is None:
        # Assuming UTC if no timezone info, adjust as necessary based on source
        sensor_data['created_at'] = sensor_data['created_at'].dt.tz_localize('UTC')
    sensor_data['created_at'] = sensor_data['created_at'].dt.tz_convert('Asia/Taipei') # Taiwan uses Asia/Taipei timezone

    # Extract date for alignment
    sensor_data['date'] = sensor_data['created_at'].dt.date
    # Convert date column to datetime objects for consistent merging later
    sensor_data['date'] = pd.to_datetime(sensor_data['date'])

    # 1.3 Data Integrity Check
    # Keep only records where status == 'complete'
    sensor_data = sensor_data[sensor_data['status'] == 'complete']

    # Ensure minimum number of sensor records per day per pond
    record_counts = sensor_data.groupby(['pond_id', 'date']).size().reset_index(name='record_count')
    valid_days = record_counts[record_counts['record_count'] >= min_records_per_day]

    # Merge to filter sensor_data
    sensor_data = pd.merge(
        sensor_data,
        valid_days[['pond_id', 'date']],
        on=['pond_id', 'date'],
        how='inner'
    )

    return sensor_data, pond_daily_logs

if __name__ == '__main__':
    # Example Usage:
    # Adjust these paths to your actual file locations
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'

    print(f"Loading sensor data from: {SENSOR_DATA_PATH}")
    print(f"Loading pond daily logs from: {POND_DAILY_LOGS_PATH}")

    processed_sensor_data, processed_pond_daily_logs = align_data(
        SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH
    )

    print("\nProcessed Sensor Data Head:")
    print(processed_sensor_data.head())
    print("\nProcessed Sensor Data Info:")
    processed_sensor_data.info()

    print("\nProcessed Pond Daily Logs Head:")
    print(processed_pond_daily_logs.head())
    print("\nProcessed Pond Daily Logs Info:")
    processed_pond_daily_logs.info()
