import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from data_utils import load_data, load_treatment_data, load_clinical_patient_data

# Page configuration
st.set_page_config(page_title="Propensity Score Matching", layout="wide")
st.title("Propensity Score Matching for Chemotherapy Cohorts")

# File paths
DATA_FILE_PATH = "msk_chord_2024/data_clinical_sample.txt"
TREATMENT_FILE_PATH = "msk_chord_2024/data_timeline_treatment.txt"
DEMOGRAPHIC_FILE_PATH = "msk_chord_2024/data_clinical_patient.txt"

# Load data
with st.spinner("Loading datasets..."):
    samples_df = load_data(DATA_FILE_PATH)
    treatment_df = load_treatment_data(TREATMENT_FILE_PATH)
    clinical_df = load_clinical_patient_data(DEMOGRAPHIC_FILE_PATH)

# Check if data loaded successfully
if samples_df is None or treatment_df is None or clinical_df is None:
    st.error("Failed to load one or more datasets. Please check file paths and try again.")
    st.stop()

# Display raw treatment data for debugging
with st.expander("Debug: Check Treatment Data"):
    st.write("Treatment Data Columns:", treatment_df.columns.tolist())
    st.write("Sample of Treatment Data:")
    st.dataframe(treatment_df.head())
    
    # Check for AGENT column specifically
    if 'AGENT' in treatment_df.columns:
        agent_counts = treatment_df['AGENT'].value_counts().head(10)
        st.write("Top 10 agents in treatment data:")
        st.dataframe(agent_counts)
    else:
        st.error("AGENT column not found in treatment data!")

# Data preprocessing
@st.cache_data
def preprocess_data(samples_df, treatment_df, clinical_df):
    # Filter treatment data for all treatment types
    # We'll skip filtering by EVENT_TYPE/SUBTYPE since this might be causing issues
    chemo_df = treatment_df.copy()
    
    # Get unique agents from the treatment data
    if 'AGENT' in chemo_df.columns:
        # Get all non-null agents
        unique_agents = chemo_df['AGENT'].dropna().unique().tolist()
        if not unique_agents:
            st.warning("No agents found in the AGENT column. Check if the column is populated.")
    else:
        st.error("AGENT column not found in treatment data")
        unique_agents = []
    
    # Merge clinical data with sample data
    merged_df = clinical_df.merge(samples_df, on='PATIENT_ID', how='inner')
    
    # Create a dictionary to hold patient treatment information
    patient_treatments = {}
    for _, row in chemo_df.iterrows():
        patient_id = row['PATIENT_ID']
        
        # Check if AGENT column exists
        if 'AGENT' not in row:
            continue
            
        agent = row['AGENT']
        
        # Skip null agents
        if pd.isna(agent):
            continue
            
        if patient_id not in patient_treatments:
            patient_treatments[patient_id] = []
            
        if agent not in patient_treatments[patient_id]:
            patient_treatments[patient_id].append(agent)
    
    # Convert the dictionary to a DataFrame for easier merging
    if patient_treatments:
        treatment_summary = pd.DataFrame({
            'PATIENT_ID': list(patient_treatments.keys()),
            'TREATMENTS': [','.join(treatments) for treatments in patient_treatments.values()]
        })
        
        # Add treatment info to the merged data
        final_df = merged_df.merge(treatment_summary, on='PATIENT_ID', how='left')
        
        # Fill NaN values in the TREATMENTS column with "No Treatment"
        final_df['TREATMENTS'] = final_df['TREATMENTS'].fillna('No Treatment')
        
        # Create treatment indicator columns
        for agent in unique_agents:
            agent_col = f'RECEIVED_{agent.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")}'
            final_df[agent_col] = final_df['TREATMENTS'].apply(
                lambda x: 1 if agent in x.split(',') else 0
            )
    else:
        final_df = merged_df.copy()
        final_df['TREATMENTS'] = 'No Treatment'
    
    # Convert categorical variables to numeric for modeling
    for col in ['GENDER', 'RACE', 'ETHNICITY', 'STAGE_HIGHEST_RECORDED']:
        if col in final_df.columns:
            final_df[col] = pd.Categorical(final_df[col])
            final_df[f'{col}_CODE'] = final_df[col].cat.codes
    
    return final_df, unique_agents

# Process data
with st.spinner("Preprocessing data..."):
    processed_data, available_agents = preprocess_data(samples_df, treatment_df, clinical_df)

# Display data summary
st.subheader("Dataset Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Patients", len(processed_data['PATIENT_ID'].unique()))
with col2:
    st.metric("Total Samples", len(processed_data))
with col3:
    st.metric("Treatment Agents", len(available_agents))

# Show all available agents for debugging
with st.expander("Available Agents"):
    st.write(sorted(available_agents))

# Check if agents are available
if not available_agents:
    st.error("No agents found in the treatment data. Please check your data and try again.")
    st.stop()

# Cohort selection
st.subheader("Select Treatment Agent for Cohort Definition")

# Display available agents for selection
selected_agent = st.selectbox(
    "Select the agent for your query cohort:",
    options=sorted(available_agents),
    index=0 if available_agents else None
)

# Make sure selected_agent is not None
if selected_agent is None:
    st.error("No agent selected. Please select an agent.")
    st.stop()

# Clean agent name for column reference
cleaned_agent = selected_agent.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
selected_agent_col = f'RECEIVED_{cleaned_agent}'

# Check if the column exists
if selected_agent_col not in processed_data.columns:
    st.error(f"Column {selected_agent_col} not found in the data. This might be due to special characters in the agent name.")
    st.stop()

# Define query and potential control cohorts
query_cohort = processed_data[processed_data[selected_agent_col] == 1].copy()
control_pool = processed_data[processed_data[selected_agent_col] == 0].copy()

# Display cohort sizes
col1, col2 = st.columns(2)
with col1:
    st.metric("Query Cohort Size", len(query_cohort))
with col2:
    st.metric("Potential Control Cohort Size", len(control_pool))

# Check if cohort sizes are sufficient
if len(query_cohort) == 0:
    st.warning(f"No patients found in the query cohort for {selected_agent}. Please select a different agent.")
    st.stop()

if len(control_pool) == 0:
    st.warning("No patients found in the control pool. Cannot perform matching.")
    st.stop()

# Covariates selection for propensity score matching
st.subheader("Select Covariates for Propensity Score Calculation")

# Define potential covariates
numerical_covariates = [col for col in ['CURRENT_AGE_DEID', 'OS_MONTHS'] if col in processed_data.columns]
categorical_covariates = [col for col in ['GENDER_CODE', 'RACE_CODE', 'ETHNICITY_CODE', 'STAGE_HIGHEST_RECORDED_CODE'] 
                         if col in processed_data.columns]

# Allow user to select covariates
selected_numerical = st.multiselect(
    "Select numerical covariates:",
    options=numerical_covariates,
    default=numerical_covariates
)

selected_categorical = st.multiselect(
    "Select categorical covariates:",
    options=categorical_covariates,
    default=categorical_covariates
)

selected_covariates = selected_numerical + selected_categorical

# Propensity score matching
def calculate_propensity_scores(df_treated, df_control, covariates):
    """Calculate propensity scores using logistic regression"""
    
    # Combine datasets for modeling
    df_treated['TREATMENT'] = 1
    df_control['TREATMENT'] = 0
    combined_df = pd.concat([df_treated, df_control], ignore_index=True)
    
    # Drop rows with missing values in covariates
    combined_df = combined_df.dropna(subset=covariates)
    
    # Check if there are enough samples after dropping missing values
    if len(combined_df[combined_df['TREATMENT'] == 1]) == 0:
        st.error("No patients in the treatment group after removing rows with missing covariate values.")
        return None, None, None
        
    if len(combined_df[combined_df['TREATMENT'] == 0]) == 0:
        st.error("No patients in the control group after removing rows with missing covariate values.")
        return None, None, None
    
    # Standardize numerical covariates
    scaler = StandardScaler()
    numerical_cols = [col for col in covariates if col in selected_numerical]
    if numerical_cols:
        combined_df[numerical_cols] = scaler.fit_transform(combined_df[numerical_cols])
    
    # Split data into features and target
    X = combined_df[covariates]
    y = combined_df['TREATMENT']
    
    # Fit logistic regression model
    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(X, y)
    except Exception as e:
        st.error(f"Error fitting propensity score model: {e}")
        return None, None, None
    
    # Calculate propensity scores
    combined_df['PROPENSITY_SCORE'] = model.predict_proba(X)[:, 1]
    
    # Separate back into treated and control groups with propensity scores
    treated_with_ps = combined_df[combined_df['TREATMENT'] == 1].copy()
    control_with_ps = combined_df[combined_df['TREATMENT'] == 0].copy()
    
    return treated_with_ps, control_with_ps, model

def match_propensity_scores(df_treated, df_control, caliper=0.2, method='nearest'):
    """Match treated patients to control patients based on propensity scores"""
    
    # Check if treated and control groups are not empty
    if df_treated is None or df_control is None:
        return None, None
        
    if len(df_treated) == 0 or len(df_control) == 0:
        st.error("Cannot perform matching: one or both cohorts are empty.")
        return None, None
    
    # Get propensity scores
    treated_ps = df_treated['PROPENSITY_SCORE'].values.reshape(-1, 1)
    control_ps = df_control['PROPENSITY_SCORE'].values.reshape(-1, 1)
    
    # Initialize list to store matched indices
    matched_control_indices = []
    matched_treated_indices = []
    
    if method == 'nearest':
        # Create a copy of the control dataframe to allow removal during matching
        control_df_copy = df_control.copy().reset_index(drop=True)
        control_ps_copy = control_ps.copy()
        
        # For each treated patient, find nearest control within caliper
        for i, ps in enumerate(treated_ps):
            # Calculate distance to all control patients
            distances = np.abs(control_ps_copy - ps)
            
            # Find nearest control within caliper
            std_ps = np.std(df_treated['PROPENSITY_SCORE']) if len(df_treated['PROPENSITY_SCORE']) > 1 else 0.1
            caliper_width = caliper * std_ps
            valid_indices = np.where(distances <= caliper_width)[0]
            
            if len(valid_indices) > 0:
                # Find closest match
                closest_idx = valid_indices[np.argmin(distances[valid_indices])]
                
                matched_treated_indices.append(i)
                matched_control_indices.append(closest_idx)
                
                # Remove the matched control to prevent reuse
                control_ps_copy = np.delete(control_ps_copy, closest_idx, axis=0)
                control_df_copy = control_df_copy.drop(closest_idx).reset_index(drop=True)
    
    # Get matched dataframes
    if matched_treated_indices and matched_control_indices:
        matched_treated = df_treated.iloc[matched_treated_indices].copy()
        
        # We need to get the original indices from df_control
        original_control = df_control.reset_index(drop=True)
        matched_control = original_control.iloc[matched_control_indices].copy()
        
        # Add match quality metrics
        match_distances = [np.abs(matched_treated['PROPENSITY_SCORE'].iloc[i] - 
                                matched_control['PROPENSITY_SCORE'].iloc[i]) 
                        for i in range(len(matched_treated))]
        
        matched_treated['MATCH_DISTANCE'] = match_distances
        matched_control['MATCH_DISTANCE'] = match_distances
        
        return matched_treated, matched_control
    else:
        return None, None

# Run propensity score matching
if st.button("Run Propensity Score Matching"):
    if not selected_covariates:
        st.error("Please select at least one covariate for matching.")
    else:
        # Set caliper based on user input
        caliper = st.slider("Caliper width (as a proportion of propensity score SD):", 0.1, 1.0, 0.2, 0.1)
        
        with st.spinner("Calculating propensity scores and matching cohorts..."):
            # Calculate propensity scores
            treated_with_ps, control_with_ps, ps_model = calculate_propensity_scores(
                query_cohort, control_pool, selected_covariates
            )
            
            if treated_with_ps is None or control_with_ps is None or ps_model is None:
                st.error("Failed to calculate propensity scores. Please check your data and selected covariates.")
            else:
                # Perform matching
                matched_treated, matched_control = match_propensity_scores(
                    treated_with_ps, control_with_ps, caliper=caliper
                )
                
                if matched_treated is None or matched_control is None or len(matched_treated) == 0:
                    st.error("Matching failed. Please try adjusting the caliper or select different covariates.")
                else:
                    # Display matching results
                    st.success(f"Successfully matched {len(matched_treated)} patients from the query cohort!")
                    
                    # Matching summary
                    st.subheader("Matching Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Query Cohort", len(query_cohort))
                    with col2:
                        st.metric("Matched Query Patients", len(matched_treated))
                    with col3:
                        match_rate = round(len(matched_treated) / len(query_cohort) * 100, 1)
                        st.metric("Match Rate", f"{match_rate}%")
                    
                    # Plot propensity score distributions
                    st.subheader("Propensity Score Distributions")
                    
                    fig = go.Figure()
                    
                    # Before matching
                    fig.add_trace(go.Histogram(
                        x=treated_with_ps['PROPENSITY_SCORE'],
                        opacity=0.7,
                        name="Query Cohort (Before Matching)",
                        marker=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=control_with_ps['PROPENSITY_SCORE'],
                        opacity=0.7,
                        name="Control Pool (Before Matching)",
                        marker=dict(color='red')
                    ))
                    
                    # After matching
                    fig.add_trace(go.Histogram(
                        x=matched_treated['PROPENSITY_SCORE'],
                        opacity=0.7,
                        name="Query Cohort (After Matching)",
                        marker=dict(color='lightblue')
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=matched_control['PROPENSITY_SCORE'],
                        opacity=0.7,
                        name="Matched Controls",
                        marker=dict(color='lightcoral')
                    ))
                    
                    fig.update_layout(
                        title="Propensity Score Distributions Before and After Matching",
                        xaxis_title="Propensity Score",
                        yaxis_title="Count",
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Covariate balance assessment
                    st.subheader("Covariate Balance Assessment")
                    
                    # Function to calculate standardized mean difference
                    def calculate_smd(x1, x2):
                        """Calculate standardized mean difference between two groups"""
                        # Check if both groups have data
                        if len(x1) == 0 or len(x2) == 0:
                            return np.nan
                            
                        mean1, mean2 = np.mean(x1), np.mean(x2)
                        var1, var2 = np.var(x1), np.var(x2)
                        
                        # Avoid division by zero
                        pooled_sd = np.sqrt((var1 + var2) / 2)
                        
                        if pooled_sd == 0:
                            return 0
                        return np.abs(mean1 - mean2) / pooled_sd
                    
                    # Calculate SMD for all covariates
                    balance_data = []
                    
                    for cov in selected_covariates:
                        # Before matching
                        smd_before = calculate_smd(
                            treated_with_ps[cov].dropna(),
                            control_with_ps[cov].dropna()
                        )
                        
                        # After matching
                        smd_after = calculate_smd(
                            matched_treated[cov].dropna(),
                            matched_control[cov].dropna()
                        )
                        
                        balance_data.append({
                            'Covariate': cov,
                            'SMD Before': smd_before,
                            'SMD After': smd_after
                        })
                    
                    balance_df = pd.DataFrame(balance_data)
                    
                    # Plot SMD values
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=balance_df['Covariate'],
                        x=balance_df['SMD Before'],
                        orientation='h',
                        name='Before Matching',
                        marker=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Bar(
                        y=balance_df['Covariate'],
                        x=balance_df['SMD After'],
                        orientation='h',
                        name='After Matching',
                        marker=dict(color='green')
                    ))
                    
                    # Add reference line at SMD = 0.1
                    fig.add_shape(
                        type="line",
                        x0=0.1, y0=-0.5,
                        x1=0.1, y1=len(balance_df) - 0.5,
                        line=dict(
                            color="gray",
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    fig.update_layout(
                        title="Standardized Mean Differences Before and After Matching",
                        xaxis_title="Standardized Mean Difference",
                        yaxis_title="Covariate",
                        barmode='group',
                        xaxis=dict(range=[0, max(
                            balance_df['SMD Before'].fillna(0).max(),
                            balance_df['SMD After'].fillna(0).max()
                        ) * 1.1 or 1])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabular balance results
                    st.write("Tabular Balance Assessment")
                    st.dataframe(balance_df.style.format({
                        'SMD Before': '{:.3f}',
                        'SMD After': '{:.3f}'
                    }).highlight_between(
                        subset=['SMD After'],
                        left=0.1,
                        right=1.0,
                        color='#ffcccc'
                    ), use_container_width=True)
                    
                    # Cohort characteristics
                    st.subheader("Cohort Characteristics")
                    
                    # Compare characteristics between matched groups
                    char_comparison = []
                    
                    for cov in selected_covariates:
                        # Skip if either group is empty for this covariate
                        if len(matched_treated[cov].dropna()) == 0 or len(matched_control[cov].dropna()) == 0:
                            continue
                            
                        treated_mean = np.mean(matched_treated[cov].dropna())
                        control_mean = np.mean(matched_control[cov].dropna())
                        
                        if cov in selected_numerical:
                            # t-test for numerical variables
                            try:
                                t_stat, p_value = stats.ttest_ind(
                                    matched_treated[cov].dropna(),
                                    matched_control[cov].dropna(),
                                    equal_var=False
                                )
                            except:
                                p_value = np.nan
                            
                            char_comparison.append({
                                'Characteristic': cov,
                                'Query Cohort': f"{treated_mean:.2f} ± {np.std(matched_treated[cov].dropna()):.2f}",
                                'Matched Controls': f"{control_mean:.2f} ± {np.std(matched_control[cov].dropna()):.2f}",
                                'p-value': p_value
                            })
                        else:
                            # Chi-square test for categorical variables
                            if len(matched_treated[cov].dropna()) > 0 and len(matched_control[cov].dropna()) > 0:
                                # Create contingency table
                                treated_counts = matched_treated[cov].value_counts().sort_index()
                                control_counts = matched_control[cov].value_counts().sort_index()
                                
                                # Ensure both have the same categories
                                all_cats = sorted(set(treated_counts.index) | set(control_counts.index))
                                treated_counts = treated_counts.reindex(all_cats, fill_value=0)
                                control_counts = control_counts.reindex(all_cats, fill_value=0)
                                
                                contingency = np.vstack([treated_counts.values, control_counts.values])
                                
                                # Run chi2 test if possible
                                try:
                                    if np.all(contingency > 0) and contingency.shape[1] > 1:
                                        chi2, p_value = stats.chi2_contingency(contingency)[:2]
                                    else:
                                        p_value = np.nan
                                except:
                                    p_value = np.nan
                            else:
                                p_value = np.nan
                            
                            char_comparison.append({
                                'Characteristic': cov,
                                'Query Cohort': f"{treated_mean:.2f}",
                                'Matched Controls': f"{control_mean:.2f}",
                                'p-value': p_value
                            })
                    
                    char_df = pd.DataFrame(char_comparison)
                    
                    # Display characteristics table
                    if not char_df.empty:
                        st.dataframe(char_df.style.format({
                            'p-value': '{:.3f}'
                        }).highlight_between(
                            subset=['p-value'],
                            left=0.0,
                            right=0.05,
                            color='#ffcccc'
                        ), use_container_width=True)
                    else:
                        st.warning("No characteristics data available for comparison.")
                    
                    # Download matched cohorts
                    st.subheader("Download Matched Cohorts")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_treated = matched_treated.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Query Cohort CSV",
                            data=csv_treated,
                            file_name=f"{cleaned_agent}_matched_query_cohort.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        csv_control = matched_control.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Matched Controls CSV",
                            data=csv_control,
                            file_name=f"{cleaned_agent}_matched_controls.csv",
                            mime="text/csv"
                        )
