# data_utils.py
import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, sep='\t', header=4)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_treatment_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        # Read the file with the existing headers (first row will be used as the column names)
        df = pd.read_csv(file_path, sep='\t')
        st.success(f"Successfully loaded treatment data from '{file_path}'")
        return df
    except Exception as e:
        st.error(f"Error loading treatment data: {e}")
        return None

@st.cache_data
def load_clinical_patient_data(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        # Read the clinical patient data while skipping the metadata lines (those starting with '#')
        df = pd.read_csv(file_path, sep='\t', comment='#')
        st.success(f"Successfully loaded clinical patient data from '{file_path}'")
        return df
    except Exception as e:
        st.error(f"Error loading clinical patient data: {e}")
        return None