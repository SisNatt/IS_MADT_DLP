import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from sklearn.preprocessing import StandardScaler

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
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')
            
            if 'Incident Type' not in df_processed.columns:
                st.error("The column 'Incident Type' is not available in the processed data.")
                st.stop()
            
            st.subheader("Step 1: Prepare Data")
            incident_types = df_processed['Incident Type'].str.get_dummies(sep=',')
            st.write("Dummy-encoded Incident Types:")
            st.dataframe(incident_types)
            
            st.subheader("Step 2: Find Frequent Itemsets")
            min_support = st.slider("Select Minimum Support", 0.01, 1.0, 0.05, step=0.01)
            frequent_itemsets = apriori(incident_types, min_support=min_support, use_colnames=True)
            
            if not frequent_itemsets.empty:
                st.write("Frequent Itemsets:")
                st.dataframe(frequent_itemsets)
                
                st.subheader("Step 3: Generate Association Rules")
                min_confidence = st.slider("Select Minimum Confidence", 0.1, 1.0, 0.5, step=0.1)
                
                # แก้ไขส่วนนี้
                rules = association_rules(
                    frequent_itemsets, 
                    metric="confidence",
                    min_threshold=min_confidence,
                    support_only=False,  # เพิ่มพารามิเตอร์นี้
                    num_itemsets=1  # เพิ่มบรรทัดนี้
                )
                
                if not rules.empty:
                    st.write("Association Rules:")
                    st.dataframe(rules)
                    
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
            else:
                st.warning("No frequent itemsets found. Try reducing the minimum support value.")
        except Exception as e:
            st.error(f"Error during pattern mining: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

# Page 5: User Behavior Analysis
elif selected == "User Behavior Analysis":
    st.title("📈 User Behavior Analysis")

    if 'processed_file' in st.session_state:
        try:
            # Load the processed file
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Check for necessary columns
            required_columns = ['Event User', 'Incident Type', 'Severity', 'Occurred (UTC)', 'Destination', 'Match_Label', 'Classification', 'Rule Set']
            missing_columns = [col for col in required_columns if col not in df_processed.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            # Step 3: Focus on Match_Label = False
            st.subheader("Step 3: Focus on Match_Label = False")
            df_processed['Match_Label'] = df_processed['Match_Label'].apply(
                lambda x: True if str(x).strip().lower() == 'true' else False
            )
            df_false = df_processed[df_processed['Match_Label'] == False]
            st.write(f"Total False records: {len(df_false)}")

            # Display False data
            with st.expander("View False Data", expanded=True):
                st.dataframe(df_false)

            # Split and Explode Classification and Rule Set
            if 'Classification' in df_false.columns and 'Rule Set' in df_false.columns:
                st.subheader("Detailed Analysis: Classification and Rule Set")

                # Split Classification and Rule Set into lists
                df_false['Classification_List'] = df_false['Classification'].str.split(',')
                df_false['Rule_Set_List'] = df_false['Rule Set'].str.split(',')

                # Explode to create individual rows for each Classification and Rule Set
                exploded_df = df_false.explode('Classification_List').explode('Rule_Set_List')

                # Remove Incident ID from display
                exploded_df = exploded_df[['Classification_List', 'Rule_Set_List', 'Severity']]
                exploded_df.columns = ['Classification', 'Rule Set', 'Severity']

                # Drop duplicate combinations
                exploded_df = exploded_df.drop_duplicates()

                # Display detailed exploded data (expanded by default)
                with st.expander("View Exploded Data", expanded=True):
                    st.dataframe(exploded_df)

                # Summarize data
                st.subheader("Summary: Classification and Severity Distribution")
                classification_summary = exploded_df.groupby(['Classification', 'Severity']).size().reset_index(name='Count')

                # Display summary data
                st.write("Summary of Classification and Severity Distribution:")
                st.dataframe(classification_summary)

                # Display numerical summary
                st.subheader("Numerical Summary:")
                for classification in classification_summary['Classification'].unique():
                    st.write(f"**Classification: {classification}**")
                    severity_data = classification_summary[classification_summary['Classification'] == classification]
                    for _, row in severity_data.iterrows():
                        severity = row['Severity']
                        count = row['Count']
                        st.write(f"- Severity '{severity}': {count} cases")
                    st.write("---")  # Separator between classifications
            else:
                st.warning("Columns 'Classification' or 'Rule Set' not found in the dataset.")


            # Step 1: Prepare Data
            st.subheader("Improved Cluster Analysis")
            cluster_data = df_processed[['Event User', 'Incident Type', 'Severity', 'Occurred (UTC)']].copy()

            # Create new features
            # 1. Count of incidents per user
            cluster_data['Incident Count'] = cluster_data.groupby('Event User')['Event User'].transform('count')

            # 2. Count of unique incident types per user
            cluster_data['Unique Incident Types'] = cluster_data.groupby('Event User')['Incident Type'].transform('nunique')

            # 3. Map severity to numeric
            severity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            cluster_data['Severity Numeric'] = cluster_data['Severity'].map(severity_mapping).fillna(0)

            # Drop duplicates and unnecessary columns
            cluster_data = cluster_data[['Event User', 'Incident Count', 'Unique Incident Types', 'Severity Numeric']].drop_duplicates()

            # Standardize the data
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_data[['Incident Count', 'Unique Incident Types', 'Severity Numeric']])

            # Step 2: Find Optimal Number of Clusters
            silhouette_scores = []
            cluster_range = range(2, 11)

            for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))

            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            st.write(f"Optimal number of clusters: {optimal_k}")

            # Plot Silhouette Scores
            silhouette_fig = px.line(
            x=list(cluster_range),
            y=silhouette_scores,
            title="Silhouette Scores for Different Number of Clusters",
            labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'}
            )
            st.plotly_chart(silhouette_fig)

            # Step 3: Perform K-Means Clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_data['Cluster Label'] = kmeans.fit_predict(scaled_features)

            # Step 4: Add Cluster Labels Back to Original Data
            df_processed = df_processed.merge(cluster_data[['Event User', 'Cluster Label']], on='Event User', how='left')

            # Display Updated DataFrame
            st.subheader("Updated DataFrame with Improved Clustering")
            st.dataframe(df_processed)

            # Step 5: Summarize Clusters
            st.subheader("Cluster Summary")
            cluster_summary = df_processed.groupby('Cluster Label').agg({
            'Event User': 'nunique',
            'Incident Type': lambda x: x.nunique(),
            'Severity': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            'Occurred (UTC)': 'count'
            }).reset_index()
            cluster_summary.columns = ['Cluster Label', 'Unique Users', 'Unique Incident Types', 'Most Common Severity', 'Total Incidents']
            st.dataframe(cluster_summary)


        except Exception as e:
            st.error(f"Error analyzing user behavior: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")

