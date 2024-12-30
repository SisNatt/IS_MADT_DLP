import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from datetime import datetime
import plotly.express as px

# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=1ueLKSgaNhvPEAjJsqad4iMcDfxsM8Nie"
DICTIONARY_FILE =  "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main Menu
selected = option_menu(
    "Main Menu", 
    ["Home - Raw Data", "Identify Incidents", "View Processed Data"], 
    icons=['house', 'search', 'bar-chart'], 
    menu_icon="cast", 
    default_index=0
)

# Page 1: Home - Raw Data
if selected == "Home - Raw Data":
    st.title("👮‍♀️👾 Data Loss Prevention reported by ML 🤖")
    st.subheader("Raw Data Overview")

    try:
        df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
        st.write(f"Total records: {len(df_raw)}")
        with st.expander("View Raw Data"):
            st.dataframe(df_raw)
    except Exception as e:
        st.error(f"Error loading raw data: {e}")

# Page 2: Identify Incidents
elif selected == "Identify Incidents":
    st.title("🔍 Identify Incidents")

    if st.button("Process Incidents"):
        try:
            df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
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

# Page 3: View Processed Data
elif selected == "View Processed Data":
    st.title("📊 View Processed Data")

    if 'processed_file' in st.session_state:
        try:
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')
            st.write(f"Total records: {len(df_processed)}")

            with st.expander("View Processed Data"):
                st.dataframe(df_processed)

            # Severity Count
            if 'Severity' in df_processed.columns:
                st.subheader("Severity Count")
                severity_count = df_processed['Severity'].value_counts().reset_index()
                severity_count.columns = ['Severity', 'Count']
                severity_fig = px.bar(severity_count, x='Severity', y='Count', color='Count', title="Severity Distribution")
                st.plotly_chart(severity_fig)
            else:
                st.error("Column 'Severity' not found in the processed dataset.")

            # Incident Type Count
            if 'Incident Type' in df_processed.columns:
                st.subheader("In
