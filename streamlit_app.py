import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
DICTIONARY_FILE = "https://drive.google.com/uc?id=1ueLKSgaNhvPEAjJsqad4iMcDfxsM8Nie"
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Title
st.title("👮‍♀️👾 Data Loss Prevention reported by ML 🤖")

# Load raw data
try:
    df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
    with st.expander("View Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df_raw)
    st.write(f"Total records: {len(df_raw)}")
except Exception as e:
    st.error(f"Error loading raw data: {e}")
    st.stop()

# Button to process incident
if st.button("Identify Incident"):
    try:
        df_dictionary = pd.read_csv(DICTIONARY_FILE, encoding='utf-8-sig')
        matching_words = set(df_dictionary['Word'].str.lower().str.strip())

        def check_evidence_match(row):
            evident_data = str(row['Evident_data']).lower().strip()
            for word in matching_words:
                if word in evident_data:
                    return 'True'
            return 'False'

        if 'Evident_data' not in df_raw.columns:
            st.error("Column 'Evident_data' not found in the dataset.")
            st.stop()

        df_raw['Match_Label'] = df_raw.apply(check_evidence_match, axis=1)

        today = datetime.now().strftime("%d%m%y")
        existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"incident_log_with_match_{today}")]
        running_number = len(existing_files) + 1
        output_file = f"{OUTPUT_DIR}/incident_log_with_match_{today}_{running_number:03d}.csv"

        df_raw.to_csv(output_file, index=False, encoding='utf-8-sig')
        st.success(f"Processed file saved as '{output_file}'")
        st.session_state['processed_file'] = output_file
    except Exception as e:
        st.error(f"Error processing incidents: {e}")

# View Data menu (after processing)
if 'processed_file' in st.session_state:
    try:
        processed_file = st.session_state['processed_file']
        df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')
        with st.expander("View Processed Data"):
            st.dataframe(df_processed)

        st.write(f"Total records: {len(df_processed)}")

        if 'Severity' in df_processed.columns:
            severity_count = df_processed['Severity'].value_counts().reset_index()
            severity_count.columns = ['Severity', 'Count']
            severity_fig = px.bar(severity_count, x='Severity', y='Count', color='Count', title="Severity Distribution")
            st.plotly_chart(severity_fig)
        else:
            st.error("Column 'Severity' not found in the processed dataset.")
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
