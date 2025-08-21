import io,re,os
from typing import List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import base64
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Data Analysis API", description="API for analyzing CSV data with natural language queries")

# Pydantic model for query input
class QueryInput(BaseModel):
    query: str

# Helper function to extract code block
def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# Query Understanding Tool
def QueryUnderstandingTool(query: str) -> bool:
    """Returns True if query requests data visualization based on keywords."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that determines if a query requests data visualization. "
                "Respond only 'True' if the query is asking for a plot, chart, graph, or any other visualization of data. "
                "Otherwise, respond with 'False'."
            )
        },
        {"role": "user", "content": query}
    ]
    
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Fallback to a known model
        temperature=0,
        max_tokens=5
    )
    response = llm.invoke(messages)
    intent_response = response.content.strip().lower()
    return intent_response == "true"

# Plot Generation Tool
def PlotGeneratorTool(cols: List[str], query: str) -> str: 
    """Generate prompt for LLM to write pandas/matplotlib code for a plot based on query and columns."""
    return f"""
Given a pandas DataFrame 'df' with columns: {', '.join(cols)}.
Write Python code using pandas and matplotlib to answer the query: "{query}"

Rules:
1. Use pandas for data manipulation and matplotlib.pyplot (import as plt) for plotting.
2. Create only one relevant plot with figsize=(8,6) and add appropriate title/labels.
3. Save the plot to a BytesIO buffer using plt.savefig(buf, format='png') and assign the buffer to 'result'.
4. Write inline code without defining functions.
5. Return your answer inside a single markdown code block that starts with ```python and ends with ```.
"""

# Non-Plot Code Generation Tool
def NonPlotGeneratorTool(cols: List[str], query: str) -> str:
    """Generate prompt for LLM to write pandas code for non-visualization queries based on query and columns."""
    return f"""
Given a pandas DataFrame 'df' with columns: {', '.join(cols)}.
Write Python code using pandas to solve the following query: "{query}"

Rules:
1. Use pandas for data manipulation.
2. Assign the final result to a variable named 'result'.
3. Return your answer inside a properly formatted markdown fenced code block that starts with ```python and ends with ```.
"""

# Code Generation Agent
def CodeGenerationAgent(query: str, df: pd.DataFrame) -> Tuple[str, bool, str]:
    """
    Selects the appropriate code generation prompt based on the query and uses the LLM to generate code.
    Returns a tuple containing (generated_code, should_plot, extra).
    """
    should_plot = QueryUnderstandingTool(query)
    
    if should_plot:
        prompt = PlotGeneratorTool(df.columns.tolist(), query)
    else:
        prompt = NonPlotGeneratorTool(df.columns.tolist(), query)
    
    system_message = (
        "Detailed thinking on. You are a Python data-analysis expert who writes clean, efficient code. "
        "Solve the given problem with optimal pandas operations. Be concise and focused. "
        "Your response must contain ONLY a properly closed ```python code block with no explanations before or after. "
        "Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Fallback to a known model
        temperature=0,
        max_tokens=512
    )
    response = llm.invoke(messages)
    
    code = extract_first_code_block(response.content)
    return code, should_plot, ""

# Execution Agent
def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool) -> Any:
    """Executes generated code in controlled environment and returns result or error message."""
    env = {"pd": pd, "df": df.copy()}  # Use copy to prevent mutation
    if should_plot:
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, env)
        result = env.get("result", None)
        if should_plot and isinstance(result, io.BytesIO):
            result.seek(0)
            return base64.b64encode(result.read()).decode('utf-8')  # Return base64-encoded image
        return result
    except Exception as exc:
        return f"Error executing code: {str(exc)}"

# Combined FastAPI Endpoint
@app.post("/analyze-csv/")
async def analyze_csv(file: UploadFile = File(...), query: str = Form(...)):
    """Upload a CSV file and process a natural language query in a single request."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file.file)
        
        # Process query
        code, should_plot, _ = CodeGenerationAgent(query, df)
        
        if not code:
            raise HTTPException(status_code=500, detail="Failed to generate code")
        
        result = ExecutionAgent(code, df, should_plot)
        
        # Construct response
        response = {"query": query, "code": code}
        
        if isinstance(result, str) and result.startswith("Error"):
            response["error"] = result
        elif should_plot:
            response["plot"] = result  # Base64-encoded image
            response["result"] = "Plot generated"
        else:
            # Handle pandas DataFrame/Series or scalar results
            if isinstance(result, (pd.DataFrame, pd.Series)):
                response["result"] = result.to_json(orient="records")
            else:
                response["result"] = str(result)
        
        response["columns"] = list(df.columns)  # Include columns for reference
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

# Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
