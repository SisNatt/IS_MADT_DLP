# User Behavior Analysis
elif selected == "User Behavior Analysis":
    st.title("📈 User Behavior Analysis")

    if 'processed_file' in st.session_state:
        try:
            processed_file = st.session_state['processed_file']
            df_processed = pd.read_csv(processed_file, encoding='utf-8-sig')

            # Existing Analysis: Analyze False Positive and Negative Cases
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

            # False Severity Analysis
            if 'Severity' in df_false.columns:
                st.subheader("False Severity Analysis")
                false_severity_count = df_false['Severity'].value_counts().reset_index()
                false_severity_count.columns = ['Severity', 'Count']
                false_severity_fig = px.bar(
                    false_severity_count,
                    x='Severity',
                    y='Count',
                    color='Count',
                    title="Severity Distribution for False Match_Label"
                )
                st.plotly_chart(false_severity_fig)

            # False Incident Type Analysis
            if 'Incident Type' in df_false.columns:
                st.subheader("False Incident Type Analysis")
                false_incident_type_count = df_false['Incident Type'].value_counts().reset_index()
                false_incident_type_count.columns = ['Incident Type', 'Count']
                false_incident_type_fig = px.bar(
                    false_incident_type_count,
                    x='Incident Type',
                    y='Count',
                    color='Count',
                    title="Incident Type Distribution for False Match_Label"
                )
                st.plotly_chart(false_incident_type_fig)

            # Frequent Words in Evident_data for False
            if 'Evident_data' in df_false.columns:
                st.subheader("Frequent Words in Evident_data (False)")
                from collections import Counter
                evident_words = df_false['Evident_data'].dropna().str.split().sum()
                word_counts = Counter(evident_words).most_common(10)
                word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

                # Display frequent words
                st.dataframe(word_df)
                word_fig = px.bar(
                    word_df,
                    x='Word',
                    y='Count',
                    color='Count',
                    title="Top Words in Evident_data for False Match_Label"
                )
                st.plotly_chart(word_fig)

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

        except Exception as e:
            st.error(f"Error during anomaly detection: {e}")
    else:
        st.warning("No processed file found. Please identify incidents first.")
