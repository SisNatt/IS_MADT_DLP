import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.express as px

# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
DICTIONARY_FILE = "https://drive.google.com/uc?id=1ueLKSgaNhvPEAjJsqad4iMcDfxsM8Nie"
OUTPUT_DIR = "./output_files"  # Directory for saving processed files
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory if not exists

# Title for the Streamlit app
st.title("👮‍♀️👾 Data Loss Prevention reported by ML 🤖")

# Load raw data
if os.path.exists(INCIDENT_FILE):
    df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
    with st.expander("View Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(df_raw)
    # Total records
    st.write(f"Total records: {len(df_raw)}")
else:
    st.error("Raw data file not found. Please ensure the file exists.")
    st.stop()

# Button to process incident
if st.button("Identify Incident"):
    if os.path.exists(DICTIONARY_FILE):
        st.write("Processing incidents...")

        # Load dictionary data
        df_dictionary = pd.read_csv(DICTIONARY_FILE, encoding='utf-8-sig')
        matching_words = set(df_dictionary['Word'].str.lower().str.strip())

        # Define matching function
        def check_evidence_match(row):
            evident_data = str(row['Evident_data']).lower().strip()
            for word in matching_words:
                if word in evident_data:
                    return 'True'
            return 'False'

        # Apply matching function and add Match_Label column
        df_raw['Match_Label'] = df_raw.apply(check_evidence_match, axis=1)

        # Generate output filename
        today = datetime.now().strftime("%d%m%y")
        existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"incident_log_with_match_{today}")]
        running_number = len(existing_files) + 1
        output_file = f"{OUTPUT_DIR}/incident_log_with_match_{today}_{running_number:03d}.csv"

        # Save processed file
        df_raw.to_csv(output_file, index=False, encoding='utf-8-sig')
        st.success(f"Processed file saved as '{output_file}'")

        # Save output file path for View Data
        st.session_state['processed_file'] = output_file

    else:
        st.error("Dictionary file not found. Please ensure the file exists.")

# View Data menu (after processing)
if 'processed_file' in st.session_state:
    st.subheader("View Data")
    processed_file = st.session_state['processed_file']

    if os.path.exists(processed_file):
        # Load processed data
        df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

        # Expander for Processed Data
        with st.expander("View Processed Data"):
            st.dataframe(df_processed)

        # Total records
        st.write(f"Total records: {len(df_processed)}")

        # Count by Severity
        st.subheader("Severity Count")
        severity_count = df_processed['Severity'].value_counts().reset_index()
        severity_count.columns = ['Severity', 'Count']

        # Interactive bar chart for Severity
        severity_fig = px.bar(severity_count, x='Severity', y='Count',
                              color='Count', color_continuous_scale='RdBu',
                              title="Severity Distribution")
        st.plotly_chart(severity_fig)

        # Count by Incident Type
        st.subheader("Incident Type Count")
        incident_type_count = df_processed['Incident Type'].value_counts().reset_index()
        incident_type_count.columns = ['Incident Type', 'Count']

        # Interactive bar chart for Incident Type
        incident_type_fig = px.bar(incident_type_count, x='Incident Type', y='Count',
                                   color='Count', color_continuous_scale='RdBu',
                                   title="Incident Type Distribution")
        st.plotly_chart(incident_type_fig)

        # Match Label Count
        st.subheader("Match Label Count")
        if 'Match_Label' in df_processed.columns:
            match_label_count = df_processed['Match_Label'].value_counts().reset_index()
            match_label_count.columns = ['Match_Label', 'Count']

            # Interactive bar chart for Match Label
            match_label_fig = px.bar(match_label_count, x='Match_Label', y='Count',
                                     color='Count', color_continuous_scale='RdBu',
                                     title="Match Label Distribution")
            st.plotly_chart(match_label_fig)
        else:
            st.error("The column 'Match_Label' does not exist in the dataset.")
