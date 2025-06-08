# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq

# Import the tools from your module
from app1 import DataInsightAgent, CodeGenerationAgent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM client factory
def get_llm():
    return ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0,
        max_tokens=512
    )

st.set_page_config(page_title="Data Analysis Assistant", layout="wide")

st.title("Data Analysis & Visualization Assistant")

# Sidebar: file uploader
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv', 'xlsx'])

if uploaded_file:
    # Read dataframe
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully.")  # NEW: feedback on load
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Show raw dataframe
    with st.expander("Raw Data Preview"):
        st.dataframe(df.head())

    # Data insights
    if st.sidebar.button("Generate Dataset Insights"):
        with st.spinner("Analyzing dataset..."):
            insights = DataInsightAgent(df)
        st.subheader("📋 Dataset Insights")
        st.write(insights)

    # User query
    st.subheader("🔍 Ask a Data Question or Request a Plot")
    query = st.text_input("Enter your analysis or visualization request:")
    if st.button("Generate Code & Result") and query:
        llm = get_llm()  # NEW: instantiate LLM for code gen
        with st.spinner("Generating code..."):
            code, should_plot = CodeGenerationAgent(query, df, llm)

        # Display generated code
        st.subheader("📝 Generated Code")
        st.code(code, language='python')

        # Execute and display result
        st.subheader("🚀 Result")
        exec_locals = {'df': df, 'plt': plt}
        try:
            exec(code, {}, exec_locals)  # CHANGED: execute generated code
            result = exec_locals.get('result', None)
            if should_plot and isinstance(result, plt.Figure):  # CHANGED: check for Figure
                st.pyplot(result)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Error executing generated code: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
