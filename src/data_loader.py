import pandas as pd
# from src.cleaner import impute_missing_sensor_values, handle_outliers

def align_data(sensor_data_path, pond_daily_logs_path, min_records_per_day=10):
    """
    Loads sensor data and pond daily logs, aligns them by date and pond_id,
    handles time zones, extracts dates, and performs data integrity checks.
    """
    # 1.1 Load Data
    sensor_data = pd.read_csv(sensor_data_path)
    pond_daily_logs = pd.read_csv(pond_daily_logs_path)

    # Convert to datetime
    sensor_data['created_at'] = pd.to_datetime(sensor_data['created_at'])
    pond_daily_logs['log_date'] = pd.to_datetime(pond_daily_logs['log_date'])

    # 1.2 Time Zone Handling & Date Extraction
    if sensor_data['created_at'].dt.tz is None:
        sensor_data['created_at'] = sensor_data['created_at'].dt.tz_localize('UTC')
    sensor_data['created_at'] = sensor_data['created_at'].dt.tz_convert('Asia/Taipei')

    # Extract date for alignment
    sensor_data['date'] = sensor_data['created_at'].dt.date
    sensor_data['date'] = pd.to_datetime(sensor_data['date'])

    # 1.3 Data Integrity Check
    sensor_data = sensor_data[sensor_data['status'] == 'complete']

    # Ensure minimum number of sensor records per day per pond
    record_counts = sensor_data.groupby(['pond_id', 'date']).size().reset_index(name='record_count')
    valid_days = record_counts[record_counts['record_count'] >= min_records_per_day]

    sensor_data = pd.merge(
        sensor_data,
        valid_days[['pond_id', 'date']],
        on=['pond_id', 'date'],
        how='inner'
    )

    return sensor_data, pond_daily_logs

# Note: impute_missing_sensor_values and handle_outliers are now imported from .cleaner

if __name__ == '__main__':
    # Example Usage:
    SENSOR_DATA_PATH = 'data/raw/sensor_data_202603291749.csv'
    POND_DAILY_LOGS_PATH = 'data/raw/pond_daily_logs_202603291749.csv'

    print(f"Loading data...")
    raw_sensor, raw_logs = align_data(SENSOR_DATA_PATH, POND_DAILY_LOGS_PATH)
    
    # print("\nApplying cleaning via imported cleaner module...")
    # cleaned_sensor = impute_missing_sensor_values(raw_sensor.copy())
    # cleaned_sensor = handle_outliers(cleaned_sensor)
    
    # print("Cleaned data head:")
    # print(cleaned_sensor.head())
