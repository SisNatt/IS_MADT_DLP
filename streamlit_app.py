import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# File paths
INCIDENT_FILE = r"D:\NIDA_MADT\IS_Project\synthetic_incident_data_#1_edit.csv"
DICTIONARY_FILE = r"D:\NIDA_MADT\IS_Project\data_dictionary2.csv"
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
# File paths
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists

# Title for the Streamlit app
st.title("👮‍♀️ Incident Log Analysis System with Pattern Mining")

# Load processed file
if 'processed_file' in st.session_state:
    processed_file = st.session_state['processed_file']
    if os.path.exists(processed_file):
        # Load data
        df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

        # Expander for Processed Data
        with st.expander("View Processed Data"):
            st.dataframe(df_processed)

        # Pattern Mining Section
        st.subheader("🔍 Incident Pattern Mining")
        st.write("Analyze frequent patterns in incident data using Apriori Algorithm.")

        # Prepare data for Apriori Algorithm
        # Convert 'Incident Type' into dummy variables for transaction analysis
        st.write("Preparing data for association rule mining...")
        incident_types = df_processed['Incident Type'].str.get_dummies(sep=',')
        st.write("Dummy-encoded Incident Types:")
        st.dataframe(incident_types)

        # Apply Apriori Algorithm
        min_support = st.slider("Select minimum support value:", 0.01, 1.0, 0.05, step=0.01)
        frequent_itemsets = apriori(incident_types, min_support=min_support, use_colnames=True)

        # Display Frequent Itemsets
        st.subheader("Frequent Itemsets")
        st.dataframe(frequent_itemsets)

        # Generate Association Rules
        min_confidence = st.slider("Select minimum confidence value:", 0.1, 1.0, 0.5, step=0.1)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # Display Association Rules
        st.subheader("Association Rules")
        st.dataframe(rules)

        # Optional Visualization of Rules
        st.subheader("Rule Visualization (Support vs Confidence)")
        if not rules.empty:
            import plotly.express as px
            fig = px.scatter(
                rules,
                x="support",
                y="confidence",
                size="lift",
                color="antecedents",
                title="Association Rule Visualization",
                labels={"antecedents": "Antecedent Patterns"}
            )
            st.plotly_chart(fig)
        else:
            st.warning("No association rules found. Adjust the minimum support or confidence values.")
    else:
        st.error("Processed file not found. Please run 'Identify Incident' first.")
else:
    st.error("No processed file available. Please process the incident data first.")
