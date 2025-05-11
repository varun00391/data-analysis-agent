
# from langchain_groq import ChatGroq
# from typing import List, Dict, Any, Tuple
# import pandas as pd

# from dotenv import load_dotenv

# load_dotenv()


# ####### Query Understanding Tool #########################

# def QueryUnderstandingTool(query:str) -> bool:
#     """Returns true if query seems to request visualization based on keywords """
#     messages = [
#         {"role": "system", "content": (
#             "You are an assistant that determines if a query requests data visualization. "
#             "Respond only 'True' if the query is asking for a plot, chart, graph, or any other visualization of data. "
#             "Otherwise, respond with 'False'."
#         )},
#         {"role": "user", "content": query}
#     ]
#     llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=0,max_tokens=5)  # Initialize the Groq model

#     response = llm.invoke(messages)
#     intent_response = response.content.strip().lower()  # Access .content directly
#     return intent_response == "true"



# ####### Plot Generation Tool #########################

# def PlotGeneratorTool(cols: List[str], query: str) -> str: 
#     """Generate a Prompt for LLM to write pandas+matplotlib code for a plot based on query and columns."""
#     return f"""
#     Given Dataframe 'df' with columns: {', '.join(cols)}
#     Write python code using pandas and matplotlib  to answer: "{query}"
    
#     Rules
     
#     1. Use pandas for data manipulation and matplotlib.pyplot (plt) for plotting .
#     2. Assign final result (Dataframw,Series,scalar or matplotlib figure) to a variable named 'result'.
#     3. Create only one relevant plot .Set 'figsize =(6,4)', and add tile/labels
#     4. Return your answer inside a single markdown fence that starts with ```Python and ends with ```.
#     """

# ####### Code Generation Tool #########################

# def CodeGenerationTool(query: str, df: pd.DataFrame) -> str:
#     """select appropriate code generation tool and gets code from LLM for user's query."""
#     should_plot = QueryUnderstandingTool(query)
#     prompt = PlotGeneratorTool(df.columns.tolist(),query) if should_plot else CodeGenerationTool(df.columns.tolist(),query)

#     messages = [
#         { "role": "system", "content": "detailed thinking. You are a Python data-analysis expert who writes clean, efficient code."
#         "Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after."
#         "Ensure your solution is correct, handles edge cases and follows best practices for data analysis"},
#         {"role": "user", "content": prompt}
#     ]

#     llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=0,max_tokens=5)  # Initialize the Groq model

#     response = llm.invoke(messages)

#     full_response  = response.choices[0].message.content
#     code = extract_first_code_block(full_response)
#     return code,should_plot, ""


# def extract_first_code_block(text: str) -> str:
#     """Extracts the first Python code block from a markdown-formatted string."""
#     start = text.find("```python")
#     if start == -1:
#         return ""
#     start += len("```python")
#     end = text.find("```", start)
#     if end == -1:
#         return ""
#     return text[start:end].strip()


import os,io, re
from dotenv import load_dotenv
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq

# Load environment variables (e.g. API keys) from .env file
load_dotenv()

####### Query Understanding Tool #########################
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
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0,
        max_tokens=5
    )
    response = llm.invoke(messages)
    # Assuming response is an object with a `content` attribute.
    intent_response = response.content.strip().lower()
    return intent_response == "true"

####### Plot Generation Tool #########################
def PlotGeneratorTool(cols: List[str], query: str) -> str: 
    """Generate prompt for LLM to write pandas/matplotlib code for a plot based on query and columns."""
    return f"""
Given a pandas DataFrame 'df' with columns: {', '.join(cols)}.
Write Python code using pandas and matplotlib to answer the query: "{query}"

Rules:
1. Use pandas for data manipulation and matplotlib.pyplot (import as plt) for plotting.
2. Create only one relevant plot with figsize=(6,4) and add appropriate title/labels.
3. Assign the final result (DataFrame, Series, scalar, or plot figure) to a variable named 'result'.
4. Return your answer inside a single markdown code block that starts with ```python and ends with ```.
"""

####### Non-Plot Code Generation Tool #########################
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

####### Code Generation Tool #########################
def CodeGenerationAgent(query: str, df: pd.DataFrame) -> Tuple[str, bool, str]:
    """
    Selects the appropriate code generation prompt based on the query and uses the LLM to generate code.
    Returns a tuple containing (generated_code, should_plot, extra).
    """
    should_plot = QueryUnderstandingTool(query)
    
    # Choose the correct prompt generator based on whether the query is for visualization.
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
    
    # Increase max_tokens to generate full code (adjust as needed)
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0,
        max_tokens=512
    )
    response = llm.invoke(messages)
    
    # Extract the full response content and then the code block from the returned markdown text.
    full_response = response.content  
    code = extract_first_code_block(full_response)
    return code, should_plot, ""


####### ExecutionAgent ##################################

def ExecutionAgent(code: str, df:pd.DataFrame, should_plot: bool):
    """Executes generated code in controlled enviroment and returns result or error message."""
    env = {"pd":pd,"df":df}
    if should_plot:
        plt.rcParams["figure.dpi"] = 100  # set default DPI for all figures
        env["plt"] = plt
        env["io"] = io
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"




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

####### Example Usage #########################
if __name__ == '__main__':
    # Create a sample DataFrame for testing
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    
    query = "Can you plot a graph for me?"
    print("QueryUnderstandingTool output:")
    print(QueryUnderstandingTool(query))
    
    code, should_plot, extra = CodeGenerationAgent(query, df)
    print("Generated Code:")
    print(code)
    print("Should Plot:", should_plot)
