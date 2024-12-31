import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

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
            required_columns = ['Event User', 'Incident Type', 'Severity', 'Occurred (UTC)', 'Match_Label', 'Classification', 'Rule Set']
            missing_columns = [col for col in required_columns if col not in df_processed.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()

            # Step 1: Prepare Data for Clustering
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

