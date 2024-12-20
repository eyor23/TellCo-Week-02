import pandas as pd

def clean_data(df):
    """
    Cleans the input DataFrame by performing basic data cleaning tasks.
    Args:
        df (pd.DataFrame): Raw DataFrame containing user session data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Fill missing values
    df = df.fillna({
        'Bearer Id': 0,
        'Start': pd.Timestamp.now(),
        'End': pd.Timestamp.now(),
        'Dur. (ms)': 0,
        'Total DL (Bytes)': 0,
        'Total UL (Bytes)': 0,
        # Add other columns as needed
    })

    # Convert time columns to datetime
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')

    # Remove rows with invalid date/time values
    df = df.dropna(subset=['Start', 'End'])

    # Ensure no negative values in session duration and data columns
    numeric_cols = ["Dur. (ms)", "HTTP DL (Bytes)", "HTTP UL (Bytes)",
                    "Social Media DL (Bytes)", "Social Media UL (Bytes)",
                    "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)",
                    "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)",
                    "Netflix DL (Bytes)", "Netflix UL (Bytes)", "Gaming DL (Bytes)",
                    "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)",
                    "Total UL (Bytes)", "Total DL (Bytes)"]
    df[numeric_cols] = df[numeric_cols].abs()

    # Ensure numerical columns are of the correct type
    numeric_cols = [
        'Bearer Id', 'Start ms', 'End ms', 'Dur. (ms)', 'IMSI',
        'MSISDN/Number', 'IMEI', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
        'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
        'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
        'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Total DL (Bytes)',
        'Total UL (Bytes)',  # Add other relevant columns
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in crucial columns
    df = df.dropna(subset=['Bearer Id', 'Start', 'End'])

    return df