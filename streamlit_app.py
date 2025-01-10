import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from collections import Counter
from sklearn.preprocessing import LabelEncoder  # Add this line


# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=15gw28r76Q_mRFoexh5d9CRHq4Hlg2iFO"
DICTIONARY_FILE = "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main Menu
with st.sidebar:
    st.title("DLP Analyst Assistant")
    st.write("""
        **DLP Analyst Assistant** is designed to streamline repetitive tasks for analysis teams, enabling faster insights and efficient policy improvements.
    """)
    selected = option_menu(
        "Main Menu",
        ["Home - Raw Data", "View Processed Data", "Pattern Mining", "User Behavior Analysis", "Anomaly Detection"],
        icons=['house', 'bar-chart', 'diagram-3', 'person', 'exclamation-triangle'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"  # Ensures the menu stays vertical
    )

# Page 1: Home - Raw Data
if selected == "Home - Raw Data":
    st.title("🛠️ DLP Analyst Assistant")
    st.write("""
        Welcome to DLP Analyst Assistant, a helpful tool designed for analyst teams
    """)
    st.subheader("Raw Data Overview")

    # Add Guideline Section
    st.subheader("📖 Guideline for using the DLP Analyst Assistant app")
    st.markdown("""
    ### **1. Purpose of the app**
    The DLP Analyst Assistant app is designed to help organizations' data analysis teams with Data Loss Prevention (DLP). It has the following functions:
    - Analyze data loss events (Incidents) to find causes and effects
    - Manage raw and processed data for in-depth analysis
    - Find duplicate data patterns (Pattern Mining) to improve DLP policies
    - Detect anomalies (Anomaly Detection) to reduce risks
    - Analyze user behavior (User Behavior Analysis) to improve inspection efficiency
    """)

    # Display raw data and other elements (unchanged from existing functionality)
    try:
        # Load raw data
        df_raw = pd.read_csv(INCIDENT_FILE, encoding='utf-8-sig')
        st.write(f"Total records: {len(df_raw)}")

        # Process 2: Data Preprocessing (New Method)
        st.subheader("Step 1: Data Preprocessing (New Method)")
        if st.button("Preprocess Data"):
            try:
                # Preprocess raw data
                df_raw.fillna('Unknown', inplace=True)
                if 'Occurred (UTC)' in df_raw.columns:
                    df_raw['Occurred (UTC)'] = pd.to_datetime(df_raw['Occurred (UTC)'])
                
                # Add new fields for encoded values
                le_severity = LabelEncoder()
                le_incident_type = LabelEncoder()
                
                # Encode and add as new columns
                if 'Severity' in df_raw.columns:
                    df_raw['Severity_Encoded'] = le_severity.fit_transform(df_raw['Severity'])
                if 'Incident Type' in df_raw.columns:
                    df_raw['Incident_Type_Encoded'] = le_incident_type.fit_transform(df_raw['Incident Type'])
                
                # Display preprocessed data
                st.write("Preprocessed Data (with Encoded Fields):")
                st.dataframe(df_raw)

                # Save preprocessed data for further analysis
                preprocessed_file = os.path.join(OUTPUT_DIR, "preprocessed_data.csv")
                df_raw.to_csv(preprocessed_file, index=False, encoding='utf-8-sig')
                st.session_state['preprocessed_file'] = preprocessed_file
                st.success("Data preprocessing completed. File saved.")

                # Optionally display mapping for encoded fields
                st.subheader("Mapping for Encoded Fields")
                st.write("**Severity Mapping**")
                st.write(dict(zip(le_severity.classes_, le_severity.transform(le_severity.classes_))))
                st.write("**Incident Type Mapping**")
                st.write(dict(zip(le_incident_type.classes_, le_incident_type.transform(le_incident_type.classes_))))

            except Exception as e:
                st.error(f"Error preprocessing raw data: {e}")
                
        # Process 1: Process Incidents (Existing Method)
        st.subheader("Step 2: Labeling Log")
        if st.button("Label Process"):
            try:
                # Load preprocessed data
                if 'preprocessed_file' in st.session_state:
                    df_raw = pd.read_csv(st.session_state['preprocessed_file'], encoding='utf-8-sig')

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

# Page 2: View Processed Data
elif selected == "View Processed Data":
    st.title("📊 View Processed Data with Match_Label Filter")

    if 'processed_file' in st.session_state:
        try:
            # Load the processed file
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')
            st.write(f"Total records: {len(df_processed)}")

            with st.expander("View Processed Data"):
                st.dataframe(df_processed)

             # เพิ่มส่วนดาวน์โหลดรายงาน
            st.subheader("Download Actionable Report")
            output_file_path = os.path.join(OUTPUT_DIR, "DLP_Insights_Report.csv")
            df_processed.to_csv(output_file_path, index=False, encoding='utf-8-sig')

            with open(output_file_path, "rb") as file:
                btn = st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="DLP_Insights_Report.csv",
                    mime="text/csv"
                )

            # Check if Match_Label exists
            if 'Match_Label' in df_processed.columns:
                st.subheader("Filter by Match_Label")
                match_label_filter = st.radio(
                    "Select Match_Label to analyze:",
                    options=['All', 'True', 'False'],
                    index=0
                )

                # Apply the matching function to ensure Match_Label is valid
                df_processed['Match_Label'] = df_processed['Match_Label'].apply(
                    lambda x: True if str(x).strip().lower() == 'true' else False
                )

                # Apply filter
                if match_label_filter == 'All':
                    df_filtered = df_processed
                elif match_label_filter == 'True':
                    df_filtered = df_processed[df_processed['Match_Label'] == True]
                elif match_label_filter == 'False':
                    df_filtered = df_processed[df_processed['Match_Label'] == False]

                # Display filtered data
                st.write(f"Filtered records: {len(df_filtered)}")
                with st.expander("View Filtered Data"):
                    st.dataframe(df_filtered)

                # Severity Count for Filtered Data
                if 'Severity' in df_filtered.columns:
                    st.subheader("Severity Count (Filtered by Match_Label)")
                    severity_count = df_filtered['Severity'].value_counts().reset_index()
                    severity_count.columns = ['Severity', 'Count']
                    severity_fig = px.bar(
                        severity_count,
                        x='Severity',
                        y='Count',
                        color='Count',
                        title=f"Severity Distribution (Filtered by Match_Label = {match_label_filter})"
                    )
                    st.plotly_chart(severity_fig)

                # Incident Type Count for Filtered Data
                if 'Incident Type' in df_filtered.columns:
                    st.subheader("Incident Type Count (Filtered by Match_Label)")
                    incident_type_count = df_filtered['Incident Type'].value_counts().reset_index()
                    incident_type_count.columns = ['Incident Type', 'Count']
                    incident_type_fig = px.bar(
                        incident_type_count,
                        x='Incident Type',
                        y='Count',
                        color='Count',
                        title=f"Incident Type Distribution (Filtered by Match_Label = {match_label_filter})"
                    )
                    st.plotly_chart(incident_type_fig)
                else:
                    st.error("Column 'Incident Type' not found in the processed dataset.")
            else:
                st.error("Column 'Match_Label' does not exist in the dataset.")
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# Pattern Mining section
elif selected == "Pattern Mining":
    st.title("🔍 Pattern Mining for Incidents")

    if 'processed_file' in st.session_state:
        try:
            # Load processed file
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            if 'Incident Type' not in df_processed.columns:
                st.error("The column 'Incident Type' is not available in the processed data.")
                st.stop()

            # Incident Trends and Patterns
            st.subheader("Incident Trends and Patterns")

            # Convert 'Occurred (UTC)' to datetime format
            df_processed['Occurred (UTC)'] = pd.to_datetime(df_processed['Occurred (UTC)'])

            # Weekly Trend Analysis
            df_processed['Week'] = df_processed['Occurred (UTC)'].dt.to_period('W').astype(str)
            weekly_trends = df_processed.groupby(['Week', 'Severity']).size().reset_index(name='Incident Count')
            overall_weekly_trends = weekly_trends.groupby('Week')['Incident Count'].sum().reset_index()

            # Combined Weekly Incident Chart
            st.subheader("Combined Weekly Incidents and Trend")
            fig_combined = px.bar(
                weekly_trends,
                x='Week',
                y='Incident Count',
                color='Severity',
                title="Weekly Incidents by Severity with Overall Trend",
                labels={'Week': 'Week', 'Incident Count': 'Number of Incidents', 'Severity': 'Severity'},
                barmode='stack'
            )

            # Add overall trend line to the combined chart
            fig_combined.add_scatter(
                x=overall_weekly_trends['Week'],
                y=overall_weekly_trends['Incident Count'],
                mode='lines+markers',
                name='Overall Trend',
                line=dict(color='black', width=2)
            )

            # Show combined chart
            st.plotly_chart(fig_combined)

            # Analysis for Weekly Trends
            st.subheader("Analysis of Weekly Trends")
            max_week = overall_weekly_trends.loc[overall_weekly_trends['Incident Count'].idxmax()]
            min_week = overall_weekly_trends.loc[overall_weekly_trends['Incident Count'].idxmin()]
            st.write(f"The week with the highest number of incidents is **{max_week['Week']}** with **{max_week['Incident Count']} incidents**.")
            st.write(f"The week with the lowest number of incidents is **{min_week['Week']}** with **{min_week['Incident Count']} incidents**.")

            # Severity and Incident Type Heatmap
            st.subheader("Severity and Incident Type Heatmap")
            heatmap_data = df_processed.groupby(['Severity', 'Incident Type']).size().reset_index(name='Count')
            fig_heatmap = px.density_heatmap(
                heatmap_data,
                x='Incident Type',
                y='Severity',
                z='Count',
                title="Heatmap of Severity vs Incident Type",
                labels={'Incident Type': 'Incident Type', 'Severity': 'Severity', 'Count': 'Number of Incidents'}
            )
            st.plotly_chart(fig_heatmap)
        
            # Recommendations Section
            st.subheader("Recommendations")
            try:
                # Generate recommendations dynamically
                most_frequent_severity = heatmap_data.groupby('Severity')['Count'].sum().idxmax()
                most_frequent_incident_type = heatmap_data.groupby('Incident Type')['Count'].sum().idxmax()
                top_combination = heatmap_data.loc[heatmap_data['Count'].idxmax()]

                st.write(f"""
                1. **Focus on {most_frequent_incident_type} Incidents**:
                   - Incident Type `{most_frequent_incident_type}` has the highest number of incidents, particularly in the `{top_combination['Severity']}` severity level.
                   - Investigate whether these incidents are routine activities or unusual behavior.

                2. **Enhance Policies for {most_frequent_severity} Incidents**:
                   - `{most_frequent_severity}` is the most frequent severity, indicating it may require stricter controls or enhanced policies.

                3. **Refine Detection Rules for Less Severe Incidents**:
                   - Focus on reducing unnecessary alerts for incidents with lower severity levels.
                """)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

         # Step 3: Grouping and Analysis
            st.subheader("Step 3: Analyzing Patterns by User and Incident Type")
            user_behavior = df_processed.groupby('Event User')['Incident Type'].value_counts()
            
            # Display results
            st.write("Pattern Analysis Grouping:")
            st.dataframe(user_behavior)

            # Save results to CSV
            pattern_csv = os.path.join(OUTPUT_DIR, "user_behavior_analysis.csv")
            user_behavior.to_csv(pattern_csv, index=True)
            st.success(f"Pattern analysis saved to {pattern_csv}")

        except Exception as e:
            st.error(f"Error during pattern mining: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")


# User Behavior Analysis
elif selected == "User Behavior Analysis":
    st.title("📊 User Behavior Analysis")

    if 'processed_file' in st.session_state:
        try:
            # Load the processed file
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Step 1: User Behavior Profile
            st.subheader("Step 1: User Behavior Profile")
            user_behavior = df_processed.groupby('Event User').agg({
                'Incident Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Severity': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Destination': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Occurred (UTC)': 'count'
            }).reset_index()
            user_behavior.columns = [
                'Event User', 'Most Frequent Incident Type',
                'Most Frequent Severity', 'Most Frequent Destination', 'Total Incidents'
            ]
            st.dataframe(user_behavior)

            # Step 2: Timeline of Top 5 Users
            st.subheader("Step 2: Timeline of Top 5 Users' Incidents (Last Month)")
            df_processed['Occurred (UTC)'] = pd.to_datetime(df_processed['Occurred (UTC)'])
            last_month = datetime.now() - pd.DateOffset(months=1)
            filtered_data = df_processed[df_processed['Occurred (UTC)'] >= last_month]
            incident_user_count = filtered_data['Event User'].value_counts().reset_index()
            incident_user_count.columns = ['Event User', 'Total Incidents']
            top_users = incident_user_count.head(5)['Event User'].tolist()
            filtered_data_top_users = filtered_data[filtered_data['Event User'].isin(top_users)]
            timeline_data = filtered_data_top_users.groupby(
                [filtered_data_top_users['Occurred (UTC)'].dt.date, 'Event User']
            ).size().reset_index(name='Incident Count')

            fig = px.line(
                timeline_data,
                x='Occurred (UTC)',
                y='Incident Count',
                color='Event User',
                title="Timeline of Incidents for Top 5 Users (Last Month)",
                labels={'Occurred (UTC)': 'Date', 'Incident Count': 'Number of Incidents'},
                markers=True
            )
            st.plotly_chart(fig)

            # Step 3: Focus on Match_Label = False
            st.subheader("Step 3: Focus on Match_Label = False")
            df_processed['Match_Label'] = df_processed['Match_Label'].apply(
                lambda x: str(x).strip().lower() == 'true'
            )
            df_false = df_processed[df_processed['Match_Label'] == False]
            st.write(f"Total False records: {len(df_false)}")

            # Show False Data in an Expander
            with st.expander("View False Data"):
                st.dataframe(df_false)

            if 'Classification' in df_false.columns:
                st.subheader("Most Frequent Classification in False Cases")
                classification_count = df_false['Classification'].value_counts().reset_index()
                classification_count.columns = ['Classification', 'Count']
                st.write(
                    f"The most frequent classification is **'{classification_count.iloc[0]['Classification']}'** "
                    f"with **{classification_count.iloc[0]['Count']}** occurrences."
                )

                # Show Classification Distribution in an Expander
                with st.expander("View Classification Distribution"):
                    st.dataframe(classification_count)
            else:
                st.warning("Column 'Classification' not found in the dataset.")

            # Step 4: Clustering for User Behavior
            st.subheader("Step 4: Clustering for User Behavior")
            features = ['Severity', 'Incident Type']
            if all(col in df_processed.columns for col in features):
                try:
                    # Encode categorical features into numeric values
                    clustering_data = df_processed[features].copy()
                    for col in features:
                        clustering_data[col] = LabelEncoder().fit_transform(clustering_data[col])

                    # Apply KMeans clustering
                    kmeans = KMeans(n_clusters=5, random_state=42)
                    df_processed['Cluster'] = kmeans.fit_predict(clustering_data)

                    # Show Clustering Results in an Expander
                    with st.expander("Clustering Results"):
                        st.dataframe(df_processed[['Event User', 'Cluster']].drop_duplicates())

                     # Analyze and Describe Clusters
                    st.subheader("Cluster Descriptions")
                    st.markdown(
                        """
                        - **Cluster 0: Low-Risk Internal Activities**
                          - Internal actions like clipboard usage and cloud file access.
                          - **Examples:** Clipboard, Cloud, Application File Access.

                        - **Cluster 1: High-Risk External Data Transfers**
                          - Activities like USB transfers and screen captures.
                          - **Examples:** Removable Storage, Screen Capture.

                        - **Cluster 2: Controlled Network Sharing**
                          - Network activities such as file sharing and printing.
                          - **Examples:** Printer, Network Share.

                        - **Cluster 3: Internet-Based Monitoring**
                          - Internet-related activities like accessing websites and capturing screens.
                          - **Examples:** Website, Screen Capture.

                        - **Cluster 4: Mixed Medium Activities**
                          - A mix of medium-based activities.
                          - **Examples:** Removable Storage, Printer, Network Share, Screen Capture.
                        """
                    )

                    # Visualization: Pie Chart
                    st.subheader("Cluster Proportions: Pie Chart")
                    fig_pie = px.pie(
                        df_processed,
                        names='Cluster',
                        title="Proportion of Users in Each Cluster",
                        hole=0.3
                    )
                    st.plotly_chart(fig_pie)

                    # Visualization: Incident Type Distribution by Cluster
                    st.subheader("Incident Type Distribution by Cluster")
                    incident_type_distribution = df_processed.groupby(['Cluster', 'Incident Type']).size().reset_index(name='Count')
                    fig_bar_incidents = px.bar(
                        incident_type_distribution,
                        x='Cluster',
                        y='Count',
                        color='Incident Type',
                        title="Incident Types by Cluster",
                        labels={'Cluster': 'Cluster', 'Count': 'Number of Incidents'},
                        barmode='stack'
                    )
                    st.plotly_chart(fig_bar_incidents)

                   
                except Exception as e:
                    st.error(f"Error during clustering: {e}")
            else:
                st.error("Required features for clustering are missing.")

        except Exception as e:
            st.error(f"Error during user behavior analysis: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# Anomaly Detection
elif selected == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    if 'processed_file' in st.session_state:
        try:
            # Load processed file from session state
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            st.subheader("Step 1: Data Preparation for Anomaly Detection")
            df_processed['Incident Count'] = df_processed.groupby('Event User')['Event User'].transform('count')
            df_processed['Unique Incident Types'] = df_processed.groupby('Event User')['Incident Type'].transform('nunique')
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            df_processed['Severity Numeric'] = df_processed['Severity'].map(severity_mapping).fillna(0)

            anomaly_features = df_processed[['Event User', 'Incident Count', 'Unique Incident Types', 'Severity Numeric']].drop_duplicates()
            st.write("Prepared Data for Anomaly Detection:")
            st.dataframe(anomaly_features)

            st.subheader("Step 2: Applying Isolation Forest")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(anomaly_features[['Incident Count', 'Unique Incident Types', 'Severity Numeric']])

            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_features['Anomaly'] = isolation_forest.fit_predict(scaled_features)

            anomaly_features['Anomaly'] = anomaly_features['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
            st.write("Anomaly Detection Results:")
            st.dataframe(anomaly_features)

            st.subheader("Step 3: Visualizing Anomalies")
            fig = px.scatter(
                anomaly_features,
                x='Incident Count',
                y='Severity Numeric',
                color='Anomaly',
                title="Anomaly Detection Visualization",
                labels={'Incident Count': 'Incident Count', 'Severity Numeric': 'Severity (Numeric)'}
            )
            st.plotly_chart(fig, use_container_width=True, key="scatter_anomalies")

            #st.subheader("Step 4: Additional Analysis - Summary of Anomalies and Normal Data")

            # Dynamically calculate and display the anomaly and normal group statistics
            #anomaly_summary = anomaly_features.groupby('Anomaly').agg({
             #   'Incident Count': ['mean', 'max', 'min'],
              #  'Unique Incident Types': ['mean', 'max', 'min'],
               # 'Severity Numeric': ['mean', 'max', 'min']
            #})

            # Flatten MultiIndex columns for easier use
            #anomaly_summary.columns = [' '.join(col).strip() for col in anomaly_summary.columns]
            #anomaly_summary = anomaly_summary.reset_index()

            # Create a dynamic table
            #st.write("### Summary Statistics for Anomaly and Normal Groups")
            #st.table(anomaly_summary)

            # Count of records for each group
            anomaly_counts = anomaly_features['Anomaly'].value_counts()
            st.write("### Count of Anomalies and Normal Group")
            st.write(f"- **Normal**: {anomaly_counts.get('Normal', 0)} Groups")
            st.write(f"- **Anomalies**: {anomaly_counts.get('Anomaly', 0)} Groups")

            # Explain findings and recommendations based on dynamic data
            #st.write("### Findings and Recommendations")
            #st.write(f"""
            #1. **Incident Count**:
             #  - Anomalies have a mean of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Anomaly', 'Incident Count mean'].values[0]:.2f}.
              # - Normal records have a mean of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Normal', 'Incident Count mean'].values[0]:.2f}.
            #2. **Severity Numeric**:
             #  - Anomalies have a mean Severity of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Anomaly', 'Severity Numeric mean'].values[0]:.2f}.
              # - Normal records have a mean Severity of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Normal', 'Severity Numeric mean'].values[0]:.2f}.
            #3. **Recommendations**:
             #  - Review high Incident Count anomalies with low Severity Numeric for potential false positives or unusual behavior.
              # - Add more contextual features to improve the quality of anomaly detection.
            #""")

            st.subheader("Step 5: Detailed Anomaly Analysis")

            # Top user with anomalies
            top_anomalous_users = df_processed[df_processed['Event User'].isin(
                anomaly_features[anomaly_features['Anomaly'] == 'Anomaly']['Event User']
            )]

            top_user = top_anomalous_users['Event User'].value_counts().idxmax()
            most_common_incident = top_anomalous_users['Incident Type'].value_counts().idxmax()
            most_common_severity = top_anomalous_users['Severity'].value_counts().idxmax()

            st.markdown(
                f"""
                ### Highlights of Anomalies
                - **Top User with Anomalies:**
                  - 🧑‍💻 **{top_user}**
                - **Most Common Incident Type in Anomalies:**
                  - 🚨 **{most_common_incident}**
                - **Most Common Severity in Anomalies:**
                  - ⚠️ **{most_common_severity}**
                """
            )

            # Visualization of Incident Type Distribution in Anomalies
            incident_type_anomalies = top_anomalous_users['Incident Type'].value_counts().reset_index()
            incident_type_anomalies.columns = ['Incident Type', 'Count']
            fig_incident_type = px.bar(
                incident_type_anomalies,
                x='Incident Type',
                y='Count',
                title="Incident Type Distribution in Anomalies",
                labels={'Incident Type': 'Incident Type', 'Count': 'Count'}
            )
            st.plotly_chart(fig_incident_type, use_container_width=True, key="bar_incident_type")

            # Visualization of Severity Distribution in Anomalies
            severity_anomalies = top_anomalous_users['Severity'].value_counts().reset_index()
            severity_anomalies.columns = ['Severity', 'Count']
            fig_severity = px.pie(
                severity_anomalies,
                names='Severity',
                values='Count',
                title="Severity Distribution in Anomalies",
                hole=0.4
            )
            st.plotly_chart(fig_severity, use_container_width=True, key="pie_severity")

        except Exception as e:
            st.error(f"Error during anomaly detection: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")
