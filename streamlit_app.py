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

            # Weekly Incidents Comparison Chart
            st.subheader("Weekly Incidents Comparison by Severity")
            fig_weekly_comparison = px.bar(
                weekly_trends,
                x='Week',
                y='Incident Count',
                color='Severity',
                title="Weekly Incident Counts by Severity",
                labels={'Week': 'Week', 'Incident Count': 'Number of Incidents', 'Severity': 'Severity'},
                barmode='group'
            )
            st.plotly_chart(fig_weekly_comparison)

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

            # Frequent Pattern Mining Section
            st.subheader("Frequent Pattern Mining")

            # Prepare data for Frequent Pattern Mining
            transaction_data = df_processed.groupby('Event User')['Incident Type'].apply(list)

            try:
                # Import required libraries
                from mlxtend.preprocessing import TransactionEncoder
                from mlxtend.frequent_patterns import apriori, association_rules

                # Apply TransactionEncoder
                te = TransactionEncoder()
                te_ary = te.fit(transaction_data).transform(transaction_data)
                df_te = pd.DataFrame(te_ary, columns=te.columns_)

                # Generate Frequent Itemsets
                frequent_itemsets = apriori(df_te, min_support=0.05, use_colnames=True)

                # Generate Association Rules
                if not frequent_itemsets.empty:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
                else:
                    rules = pd.DataFrame()

                # Display Frequent Itemsets
                st.write("### Frequent Itemsets")
                if not frequent_itemsets.empty:
                    st.dataframe(frequent_itemsets)
                else:
                    st.warning("No frequent itemsets found. Try reducing the `min_support` threshold.")

                # Display Association Rules
                st.write("### Association Rules")
                if not rules.empty:
                    st.dataframe(rules)

                    # Visualize Association Rules
                    fig_rules = px.scatter(
                        rules,
                        x='confidence',
                        y='lift',
                        size='support',
                        color=rules['antecedents'].apply(lambda x: ', '.join(list(x))),
                        title="Association Rules Visualization",
                        labels={'confidence': 'Confidence', 'lift': 'Lift', 'support': 'Support'}
                    )
                    st.plotly_chart(fig_rules)
                else:
                    st.warning("No association rules found. Try adjusting the `min_threshold` for confidence.")

            except Exception as e:
                st.error(f"Error in Frequent Pattern Mining: {e}")

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

            # Added Analysis: User Behavior Profile and Clustering
            # Step 1: User Behavior Profile
            st.subheader("Step 1: User Behavior Profile")
            user_behavior = df_processed.groupby('Event User').agg({
                'Incident Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Severity': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Destination': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                'Occurred (UTC)': 'count'
            }).reset_index()
            user_behavior.columns = ['Event User', 'Most Frequent Incident Type', 'Most Frequent Severity',
                                     'Most Frequent Destination', 'Total Incidents']
            st.dataframe(user_behavior)

            # Step 2: Timeline of Top 5 Users
            st.subheader("Timeline of Top 5 Users' Incidents (Last Month)")
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

            # Plot timeline
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
                lambda x: True if str(x).strip().lower() == 'true' else False
            )
            df_false = df_processed[df_processed['Match_Label'] == False]
            st.write(f"Total False records: {len(df_false)}")

            # Display False data
            with st.expander("View False Data"):
                st.dataframe(df_false)

            # Analyze Classification with highest frequency
            if 'Classification' in df_false.columns:
                st.subheader("Most Frequent Classification in False Cases")
                classification_count = df_false['Classification'].value_counts().reset_index()
                classification_count.columns = ['Classification', 'Count']
                most_frequent_classification = classification_count.iloc[0]  # Get the top row
                st.write(f"The most frequent classification is **'{most_frequent_classification['Classification']}'** "
                f"with **{most_frequent_classification['Count']}** occurrences.")

                # Optional: Display the full classification count table
                st.write("Classification Distribution:")
                st.dataframe(classification_count)
            else:
                st.warning("Column 'Classification' not found in the dataset.")


            # Frequent Words in Evident_data for False
            #if 'Evident_data' in df_false.columns:
                #st.subheader("Frequent Words in Evident_data (False)")
                #from collections import Counter
                #evident_words = df_false['Evident_data'].dropna().str.split().sum()
                #word_counts = Counter(evident_words).most_common(10)
                #word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

                # Display frequent words
                #st.dataframe(word_df)
                #word_fig = px.bar(
                    #word_df,
                    #x='Word',
                    #y='Count',
                    #color='Count',
                    #title="Top Words in Evident_data for False Match_Label"
                #)
                #st.plotly_chart(word_fig)

            # Clustering Analysis
            st.subheader("Clustering Incident Data")
            cluster_data = df_processed[['Event User', 'Incident Type', 'Severity', 'Occurred (UTC)']].copy()

            # Create new features
            cluster_data['Incident Count'] = cluster_data.groupby('Event User')['Event User'].transform('count')
            cluster_data['Unique Incident Types'] = cluster_data.groupby('Event User')['Incident Type'].transform('nunique')
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            cluster_data['Severity Numeric'] = cluster_data['Severity'].map(severity_mapping).fillna(0)

            # Drop duplicates and unnecessary columns
            cluster_data = cluster_data[['Event User', 'Incident Count', 'Unique Incident Types', 'Severity Numeric']].drop_duplicates()

            # Standardize the data
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_data[['Incident Count', 'Unique Incident Types', 'Severity Numeric']])

            # Perform K-Means Clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_data['Cluster Label'] = kmeans.fit_predict(scaled_features)

            # Add Cluster Labels Back to Original Data
            df_processed = df_processed.merge(cluster_data[['Event User', 'Cluster Label']], on='Event User', how='left')

            # Step 2: Create Incident Patterns for Visualization
            st.subheader("Visualization: Incident Patterns by Cluster")
            incident_patterns = df_processed.groupby(['Cluster Label', 'Incident Type']).size().reset_index(name='Count')

            # Display the DataFrame for verification
            st.write("Incident Patterns by Cluster and Incident Type:")
            st.dataframe(incident_patterns)

            # Step 3: Plot the Cluster Analysis
            st.subheader("Cluster Analysis: Incident Count by Incident Type")

            # Plot the Bar Chart
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=incident_patterns, 
                x='Cluster Label',          # Use Cluster Label as X-axis
                y='Count',                  # Use Count as Y-axis
                hue='Incident Type',        # Color by Incident Type
                palette='Set2'              # Use a color palette for better visualization
            )
            # อธิบาย Cluster Analysis
            st.subheader("Cluster Analysis Insights")
            cluster_summary = cluster_data.groupby('Cluster Label').agg({
            'Incident Count': 'mean',
            'Severity Numeric': 'mean',
            }).reset_index()
            cluster_summary['Risk Level'] = cluster_summary['Severity Numeric'].apply(
            lambda x: 'High' if x > 3 else 'Medium' if x > 2 else 'Low'
            )
            st.write("Cluster Summary and Risk Level:")
            st.dataframe(cluster_summary)


            # Customize the Chart
            plt.title("Incident Count by Cluster and Incident Type")
            plt.xlabel("Cluster")
            plt.ylabel("Incident Count")
            plt.legend(title="Incident Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(plt.gcf())  # Show the chart in Streamlit

        except Exception as e:
            st.error(f"Error analyzing user behavior: {e}")
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
            st.plotly_chart(fig)

            # Step 4: Displaying Dynamic Analysis Results
            st.subheader("Step 4: Additional Analysis - Summary of Anomalies and Normal Data")

            # Dynamically calculate and display the anomaly and normal group statistics
            anomaly_summary = anomaly_features.groupby('Anomaly').agg({
                'Incident Count': ['mean', 'max', 'min'],
                'Unique Incident Types': ['mean', 'max', 'min'],
                'Severity Numeric': ['mean', 'max', 'min']
            })

            # Flatten MultiIndex columns for easier use
            anomaly_summary.columns = [' '.join(col).strip() for col in anomaly_summary.columns]
            anomaly_summary = anomaly_summary.reset_index()

            # Create a dynamic table
            st.write("### Summary Statistics for Anomaly and Normal Groups")
            st.table(anomaly_summary)

            # Count of records for each group
            anomaly_counts = anomaly_features['Anomaly'].value_counts()
            st.write("### Count of Anomalies and Normal Records")
            st.write(f"- **Normal**: {anomaly_counts.get('Normal', 0)} records")
            st.write(f"- **Anomalies**: {anomaly_counts.get('Anomaly', 0)} records")

            # Explain findings and recommendations based on dynamic data
            st.write("### Findings and Recommendations")
            st.write(f"""
            1. **Incident Count**:
               - Anomalies have a mean of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Anomaly', 'Incident Count mean'].values[0]:.2f}.
               - Normal records have a mean of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Normal', 'Incident Count mean'].values[0]:.2f}.
            2. **Severity Numeric**:
               - Anomalies have a mean Severity of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Anomaly', 'Severity Numeric mean'].values[0]:.2f}.
               - Normal records have a mean Severity of {anomaly_summary.loc[anomaly_summary['Anomaly'] == 'Normal', 'Severity Numeric mean'].values[0]:.2f}.
            3. **Recommendations**:
               - Review high Incident Count anomalies with low Severity Numeric for potential false positives or unusual behavior.
               - Add more contextual features to improve the quality of anomaly detection.
            """)

        except Exception as e:
            st.error(f"Error during anomaly detection: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

