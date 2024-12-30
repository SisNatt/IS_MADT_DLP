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
    ["Home - Raw Data", "View Processed Data", "Pattern Mining", "User Behavior Analysis"], 
    icons=['house', 'bar-chart', 'diagram-3', 'person'], 
    menu_icon="cast", 
    default_index=0
)

# Page 1: Home - Raw Data
if selected == "Home - Raw Data":
    st.title("👮‍♀️👾 Data Loss Prevention reported by ML 🤖")
    st.subheader("Raw Data Overview")

    try:
        # Load raw data
        df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
        st.write(f"Total records: {len(df_raw)}")

        # Display raw data in expander
        with st.expander("View Raw Data"):
            st.dataframe(df_raw)

        # Add a button to process incidents
        if st.button("Process Incidents"):
            try:
                # Load dictionary
                df_dictionary = pd.read_csv(DICTIONARY_FILE, encoding='utf-8-sig')

                # Define matching words
                matching_words = set(df_dictionary['Word'].str.lower().str.strip())

                def check_evidence_match(row):
                    """Check if words in 'Evident_data' match the dictionary."""
                    evident_data = str(row['Evident_data']).lower().strip()
                    for word in matching_words:
                        if word in evident_data:
                            return 'True'
                    return 'False'

                # Validate 'Evident_data' column
                if 'Evident_data' not in df_raw.columns:
                    st.error("Column 'Evident_data' not found in the dataset.")
                    st.stop()

                # Apply the matching function
                df_raw['Match_Label'] = df_raw.apply(check_evidence_match, axis=1)

                # Save processed file
                today = datetime.now().strftime("%d%m%y")
                existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"incident_log_with_match_{today}")]
                running_number = len(existing_files) + 1
                output_file = f"{OUTPUT_DIR}/incident_log_with_match_{today}_{running_number:03d}.csv"
                df_raw.to_csv(output_file, index=False, encoding='utf-8-sig')

                # Save processed file to session state
                st.session_state['processed_file'] = output_file
                st.success(f"Processed file saved as '{output_file}'")
            except Exception as e:
                st.error(f"Error processing incidents: {e}")

    except Exception as e:
        st.error(f"Error loading raw data: {e}")

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

# Page: User Behavior Analysis
if selected == "User Behavior Analysis":
    st.title("👤 User Behavior Analysis")

    if 'processed_file' in st.session_state:
        try:
            # Load processed data
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Step 1: ตรวจสอบและแปลงคอลัมน์ Severity เป็นตัวเลข
            if 'Severity' in df_processed.columns:
                try:
                    df_processed['Severity'] = pd.to_numeric(df_processed['Severity'], errors='coerce')
                    df_processed['Severity'] = df_processed['Severity'].fillna(df_processed['Severity'].mean())
                except Exception as e:
                    st.error(f"Error converting Severity column: {e}")
            else:
                st.error("Column 'Severity' not found in the dataset.")
                st.stop()

            # Step 2: ตรวจสอบคอลัมน์วันที่และเวลา
            if 'Occurred (UTC)' in df_processed.columns:
                date_column = 'Occurred (UTC)'
            elif 'Time' in df_processed.columns:
                date_column = 'Time'
            else:
                st.error("No valid date or time column found in the dataset.")
                st.stop()

            # Step 3: วิเคราะห์พฤติกรรมของ Event User
            st.subheader("Step 1: User Behavior Analysis")
            user_behavior = df_processed.groupby('Event User').agg({
                'Incident Type': 'count',
                'Severity': 'mean',
                date_column: ['min', 'max']
            }).reset_index()

            user_behavior.columns = ['Event User', 'Total Incidents', 'Average Severity', 'First Access', 'Last Access']

            # เลือก Top 10 Event Users
            user_behavior = user_behavior.nlargest(10, 'Total Incidents')
            st.write("Top 10 Event User Behavior Overview:")
            st.dataframe(user_behavior)

            # Visualize Top 10 User Behavior
            user_incident_fig = px.bar(
                user_behavior,
                x='Event User',
                y='Total Incidents',
                color='Average Severity',
                title="Top 10 User Behavior: Total Incidents and Average Severity",
                labels={"Total Incidents": "Number of Incidents", "Average Severity": "Severity"}
            )
            st.plotly_chart(user_incident_fig)

          # Step 4: Anomaly Detection for Event Users
        st.subheader("Step 2: Detect Anomalous User Behavior")

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # เตรียมข้อมูลสำหรับ Anomaly Detection
        anomaly_data = user_behavior[['Total Incidents', 'Average Severity']]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(anomaly_data)

        # ใช้ Isolation Forest
        isolation_forest = IsolationForest(random_state=42)
        isolation_forest.fit(scaled_data)  # เรียกใช้ fit ก่อน

        # คำนวณ Anomaly Score และคาดการณ์ Anomaly
        user_behavior['Anomaly Score'] = isolation_forest.decision_function(scaled_data)
        user_behavior['Anomaly'] = isolation_forest.predict(scaled_data)

        # แปลง Anomaly Score ให้เป็นค่าบวก
        user_behavior['Anomaly Score (Positive)'] = user_behavior['Anomaly Score'].abs()

        # เลือก Top 10 Anomalous Users
        anomalies = user_behavior[user_behavior['Anomaly'] == -1].nlargest(10, 'Anomaly Score')
        st.write("Top 10 Anomalous Users:")
        st.dataframe(anomalies)

        # Visualize Top 10 Anomalies
        anomaly_fig = px.scatter(
        anomalies,
        x='Total Incidents',
        y='Average Severity',
        size='Anomaly Score (Positive)',  # ใช้ค่าบวกสำหรับ size
        color='Anomaly Score',
        title="Top 10 Anomalous User Behavior",
        labels={"Total Incidents": "Number of Incidents", "Average Severity": "Severity"}
        )
        st.plotly_chart(anomaly_fig)

            # Step 5: Behavior Clustering
            st.subheader("Step 3: Behavior Clustering")
            from sklearn.cluster import KMeans

            # Apply K-Means Clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            user_behavior['Cluster'] = kmeans.fit_predict(scaled_data)

            # Visualize Clusters
            cluster_fig = px.scatter(
                user_behavior,
                x='Total Incidents',
                y='Average Severity',
                color='Cluster',
                title="User Behavior Clustering: 5 Groups",
                labels={"Cluster": "Cluster Group"}
            )
            st.plotly_chart(cluster_fig)

            # อธิบายกลุ่มพฤติกรรม
            st.markdown("### Cluster Characteristics")
            clusters_description = {
                0: "Clipboard & Cloud Users: Copying data to the clipboard and using cloud applications.",
                1: "High-Risk Storage & Print: High usage of removable storage or printing data, indicating risk of data being taken outside the organization.",
                2: "Mixed Activity Users: Engaging in a variety of incident types, such as removable storage, screen capturing, and web access.",
                3: "File Access & Cloud Users: Accessing files through applications with lower risk but still requiring monitoring.",
                4: "Screen Capture & Web Users: Focusing on screen capturing and web usage, which may indicate data monitoring concerns."
            }

            for cluster, description in clusters_description.items():
                st.markdown(f"**Cluster {cluster}:** {description}")

        except Exception as e:
            st.error(f"Error analyzing user behavior: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")



