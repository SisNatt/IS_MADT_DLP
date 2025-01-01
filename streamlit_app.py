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

# File paths
INCIDENT_FILE = "https://drive.google.com/uc?id=1ueLKSgaNhvPEAjJsqad4iMcDfxsM8Nie"
DICTIONARY_FILE = "https://drive.google.com/uc?id=1RF7pbtpx9OjiqKhASwjxmpJfHHP9ulkv"
OUTPUT_DIR = "./output_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main Menu
selected = option_menu(
    "Main Menu", 
    ["Home - Raw Data", "View Processed Data", "Pattern Mining", "User Behavior Analysis", "Anomaly Detection"], 
    icons=['house', 'bar-chart', 'diagram-3', 'person', 'bell-exclamation'], 
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
            weekly_trends = df_processed.groupby('Week').size().reset_index(name='Incident Count')
            fig_trend = px.line(
                weekly_trends,
                x='Week',
                y='Incident Count',
                title="Weekly Trend of Incidents",
                labels={'Week': 'Week', 'Incident Count': 'Number of Incidents'}
            )
            st.plotly_chart(fig_trend)

            # Severity and Incident Type Heatmap
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

        except Exception as e:
            st.error(f"Error during pattern mining: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# User Behavior Analysis
elif selected == "User Behavior Analysis":
    st.title("📈 User Behavior Analysis")

    if 'processed_file' in st.session_state:
        try:
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            st.subheader("Analyze False Positive and Negative Cases")
            false_cases = df_processed[df_processed['Match_Label'] == False]
            st.write(f"Total False Cases: {len(false_cases)}")

            # Severity distribution
            if 'Severity' in false_cases.columns:
                severity_count = false_cases['Severity'].value_counts().reset_index()
                severity_count.columns = ['Severity', 'Count']
                fig = px.bar(severity_count, x='Severity', y='Count', title="Severity Distribution in False Cases")
                st.plotly_chart(fig)

            # Frequent words in 'Evident_data'
            if 'Evident_data' in false_cases.columns:
                evident_words = false_cases['Evident_data'].dropna().str.split().sum()
                word_counts = Counter(evident_words).most_common(10)
                word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                st.write("Frequent Words in False Cases:")
                st.dataframe(word_df)
                word_fig = px.bar(word_df, x='Word', y='Count', title="Frequent Words in Evident_data")
                st.plotly_chart(word_fig)

            st.subheader("Policy Recommendations")
            recommendations = false_cases.groupby('Incident Type').size().reset_index(name='False Count')
            recommendations['Recommendation'] = recommendations['Incident Type'].apply(
                lambda x: f"Adjust rules for '{x}' to reduce False Cases"
            )
            st.dataframe(recommendations)

        except Exception as e:
            st.error(f"Error analyzing user behavior: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# Menu 6: Anomaly Detection
elif selected == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    if 'processed_file' in st.session_state:
        try:
            # Load the processed file
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Check for necessary columns
            required_columns = ['Event User', 'Incident Type', 'Severity', 'Occurred (UTC)', 'Classification', 'Rule Set', 'Match_Label']
            missing_columns = [col for col in required_columns if col not in df_processed.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            # Step 1: Data Preparation for Anomaly Detection
            st.subheader("Step 1: Data Preparation")

            # Create new features
            df_processed['Incident Count'] = df_processed.groupby('Event User')['Event User'].transform('count')
            df_processed['Unique Incident Types'] = df_processed.groupby('Event User')['Incident Type'].transform('nunique')
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            df_processed['Severity Numeric'] = df_processed['Severity'].map(severity_mapping).fillna(0)

            # Prepare data for modeling
            anomaly_features = df_processed[['Event User', 'Incident Count', 'Unique Incident Types', 'Severity Numeric']].drop_duplicates()
            st.write("Prepared Data for Anomaly Detection:")
            st.dataframe(anomaly_features)

            # Step 2: Apply Isolation Forest for Anomaly Detection
            st.subheader("Step 2: Anomaly Detection Using Isolation Forest")

            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            # Normalize the features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(anomaly_features[['Incident Count', 'Unique Incident Types', 'Severity Numeric']])

            # Train Isolation Forest
            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_features['Anomaly'] = isolation_forest.fit_predict(scaled_features)

            # Mark anomalies (-1 as anomalies, 1 as normal)
            anomaly_features['Anomaly'] = anomaly_features['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
            st.write("Anomaly Detection Results:")
            st.dataframe(anomaly_features)

            # Step 3: Visualize Anomalies
            st.subheader("Step 3: Anomaly Visualization")

            import plotly.express as px

            # Scatter plot for anomalies
            fig = px.scatter(
                anomaly_features,
                x='Incident Count',
                y='Severity Numeric',
                color='Anomaly',
                title="Anomaly Detection Visualization",
                labels={'Incident Count': 'Incident Count', 'Severity Numeric': 'Severity (Numeric)'}
            )
            st.plotly_chart(fig)

            # Step 4: Analyze Anomalous Data
            st.subheader("Step 4: Analysis of Anomalous Data")

            anomalous_data = anomaly_features[anomaly_features['Anomaly'] == 'Anomaly']
            if anomalous_data.empty:
                st.warning("No anomalies detected. Try adjusting the contamination parameter.")
            else:
                st.write("Anomalous Data Summary:")
                st.dataframe(anomalous_data)

                # Identify associated users and incidents
                anomalous_users = df_processed[df_processed['Event User'].isin(anomalous_data['Event User'])]
                if anomalous_users.empty:
                    st.warning("No matching users found for the detected anomalies.")
                else:
                    st.write("Details of Users with Anomalies:")
                    st.dataframe(
                        anomalous_users.drop(columns=['Incident ID'])  # Exclude 'Incident ID' column from the display
                    )

                    # Step 5: Top 10 Anomalous Users Visualization
                    st.subheader("Top 10 Anomalous Users by Incident Count")
                    top_anomalous_users = anomalous_users.groupby('Event User')['Incident Count'].max().reset_index()
                    top_anomalous_users = top_anomalous_users.sort_values(by='Incident Count', ascending=False).head(10)

                    # Visualization for Top 10 Anomalous Users
                    top_users_fig = px.bar(
                        top_anomalous_users,
                        x='Event User',
                        y='Incident Count',
                        color='Incident Count',
                        title="Top 10 Anomalous Users by Incident Count",
                        labels={'Incident Count': 'Incident Count', 'Event User': 'Event User'}
                    )
                    st.plotly_chart(top_users_fig)

        except Exception as e:
            st.error(f"Error analyzing anomalies: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")"
