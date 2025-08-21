import streamlit as st
import requests
import pandas as pd
import base64
from io import BytesIO

# Streamlit app configuration
st.set_page_config(page_title="Data Analysis App", page_icon="📊", layout="wide")

# Title and description
st.title("📊 Data Analysis App")
st.markdown("""
Upload a CSV file and enter a natural language query to analyze your data. 
The app will generate Python code using pandas and matplotlib, execute it, and display the results or a plot.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload a CSV file to analyze.")

# Query input
query = st.text_input("Enter your query", placeholder="e.g., 'Plot sales distribution' or 'Average age by gender'")

# Button to submit
if st.button("Analyze", disabled=not (uploaded_file and query)):
    with st.spinner("Processing your query..."):
        try:
            # Prepare the POST request to the FastAPI endpoint
            url = "http://localhost:8000/analyze-csv/"
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            data = {"query": query}
            
            # Send request
            response = requests.post(url, files=files, data=data)
            
            # Check for HTTP errors
            if response.status_code != 200:
                st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            else:
                # Parse JSON response
                result = response.json()
                
                # Display query
                st.subheader("Query")
                st.write(result.get("query", "No query returned"))
                
                # Display columns
                st.subheader("CSV Columns")
                columns = result.get("columns", [])
                if columns:
                    st.write(", ".join(columns))
                else:
                    st.write("No columns returned")
                
                # Display generated code
                st.subheader("Generated Code")
                code = result.get("code", "")
                if code:
                    st.code(code, language="python")
                else:
                    st.warning("No code generated")
                
                # Display error if present
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                
                # Display result or plot
                elif "plot" in result:
                    st.subheader("Plot")
                    try:
                        # Decode base64 plot and display as image
                        plot_data = result["plot"]
                        st.image(f"data:image/png;base64,{plot_data}")
                        st.write("Result: Plot generated")
                    except Exception as e:
                        st.error(f"Error displaying plot: {str(e)}")
                else:
                    st.subheader("Result")
                    result_data = result.get("result", "No result returned")
                    try:
                        # Try to parse result as JSON (for DataFrame/Series)
                        import json
                        parsed_result = json.loads(result_data)
                        if isinstance(parsed_result, list):
                            df = pd.DataFrame(parsed_result)
                            st.dataframe(df)
                        else:
                            st.write(result_data)
                    except json.JSONDecodeError:
                        # Handle scalar results
                        st.write(result_data)
        
        except requests.RequestException as e:
            st.error(f"Failed to connect to API: {str(e)}")
            st.info("Ensure the FastAPI server is running at http://localhost:8000")

# Footer
st.markdown("---")
st.markdown("Powered by FastAPI and Streamlit | Developed for data analysis with natural language queries")