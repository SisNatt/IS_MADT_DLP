import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=1ueLKSgaNhvPEAjJsqad4iMcDfxsM8Nie"
DICTIONARY_FILE = "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main Menu
selected = option_menu(
    "Main Menu", 
    ["Home - Raw Data", "Identify Incidents", "View Processed Data", "Pattern Mining"], 
    icons=['house', 'search', 'bar-chart', 'diagram-3'], 
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
                st.subheader("Incident Type Count")
                incident_type_count = df_processed['Incident Type'].value_counts().reset_index()
                incident_type_count.columns = ['Incident Type', 'Count']
                incident_type_fig = px.bar(incident_type_count, x='Incident Type', y='Count',
                                           color='Count', color_continuous_scale='RdBu',
                                           title="Incident Type Distribution")
                st.plotly_chart(incident_type_fig)
            else:
                st.error("Column 'Incident Type' not found in the processed dataset.")

            # Match Label Count
            if 'Match_Label' in df_processed.columns:
                st.subheader("Match Label Count")
                match_label_count = df_processed['Match_Label'].value_counts().reset_index()
                match_label_count.columns = ['Match_Label', 'Count']
                match_label_fig = px.bar(match_label_count, x='Match_Label', y='Count',
                                         color='Count', color_continuous_scale='RdBu',
                                         title="Match Label Distribution")
                st.plotly_chart(match_label_fig)
            else:
                st.error("The column 'Match_Label' does not exist in the dataset.")
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# Page 4: Pattern Mining
elif selected == "Pattern Mining":
    st.title("🔍 Pattern Mining for Incidents")

    if 'processed_file' in st.session_state:
        try:
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Check for Incident Type column
            if 'Incident Type' not in df_processed.columns:
                st.error("The column 'Incident Type' is not available in the processed data.")
                st.stop()

            # Prepare data for Apriori
            st.subheader("Step 1: Prepare Data")
            incident_types = df_processed['Incident Type'].str.get_dummies(sep=',')
            st.write("Dummy-encoded Incident Types:")
            st.dataframe(incident_types)

            # Apriori Algorithm
            st.subheader("Step 2: Find Frequent Itemsets")
            min_support = st.slider("Select Minimum Support", 0.01, 1.0, 0.05, step=0.01)
            frequent_itemsets = apriori(incident_types, min_support=min_support, use_colnames=True)

            if not frequent_itemsets.empty:
                st.write("Frequent Itemsets:")
                st.dataframe(frequent_itemsets)
            else:
                st.warning("No frequent itemsets found. Try reducing the minimum support value.")

            # Association Rules
            if not frequent_itemsets.empty:
                st.subheader("Step 3: Generate Association Rules")
                min_confidence = st.slider("Select Minimum Confidence", 0.1, 1.0, 0.5, step=0.1)
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                if 'rules' in locals() and not rules.empty:
                    st.write("Association Rules:")
                    st.dataframe(rules)

                    # Explanation
                    st.markdown("""
                    - **Association Rules** แสดงกฎความสัมพันธ์ระหว่างเหตุการณ์
                    - ค่า Confidence บ่งบอกความน่าจะเป็นที่เหตุการณ์ปลายทางจะเกิดขึ้นเมื่อมีเหตุการณ์ต้นทาง
                    - ค่า Lift > 1 บ่งบอกถึงความสัมพันธ์ที่มีความแข็งแรง
                    """)

                    # Visualization
                    st.subheader("Step 4: Visualization of Rules")
                    fig = px.scatter(
                        rules,
                        x="support",
                        y="confidence",
                        size="lift",
                        color="antecedents",
                        title="Support vs Confidence",
                        labels={"antecedents": "Antecedent Patterns"}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("No association rules found. Try reducing the minimum confidence value.")
        except Exception as e:
            st.error(f"Error during pattern mining: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")
