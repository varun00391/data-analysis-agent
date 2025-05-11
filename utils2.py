
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
from typing import List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from langchain_groq import ChatGroq
import streamlit as st

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
        exec(code,env)   #exec(code, {}, env)
        return env.get("result", None)
    except Exception as exc:
        return f"Error executing code: {exc}"



############### ReasoningCurator Tool #################

def ReasoningCurator(query:str, result:Any) -> str:
    """Build and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result,str) and result.startswith("Error executing code")
    is_plot = isinstance(result,(Figure,Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result,Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result,Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)[:300]
    
    if is_plot:
        prompt= f'''
        The user asked: "{query}".
        Below is the description of the plot result:
        {desc}
        Explain in 2-3 concise sentences what the chart shows (no code talk). 
        '''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain in 2-3 concise sentences what this tells about the data (no mention of charts).
        '''
    return prompt

############### ReasoningAgent (streaming) ########################

def ReasoningAgent(query:str, result:Any):
    """ Streams the LLM's reasoning about the result (plot or value) and extracts model 'thinking' and final explanation."""
    prompt = ReasoningCurator(query,result)
    is_error = isinstance(result,str) and result.startswith("Error executing code")
    is_code = isinstance(result,(Figure,Axes))

    # streaming llm call

    messages = [
        {"role": "system", "content": "Detailed thinking on. You are an insightful senior data analyst"},
        {"role": "user", "content": prompt}
    ]
    # response = 
    # Increase max_tokens to generate full code (adjust as needed)
    llm = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0,
        max_tokens=1024
    )
    
    # FIX: Wrap the streaming call in a try/except block to ensure 'response' is defined.
    try:
        response = llm.invoke(messages, stream=True)
    except Exception as e:
        response = []
    
    # FIX: Ensure response is iterable; if not, wrap it in a list.
    if not hasattr(response, '__iter__'):
        response = [response]

    ## Stream and display thinking
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        # Try to extract token content. We support both object attribute and dict cases.
        if hasattr(chunk, "choices"):
            token = chunk.choices[0].delta.content
        elif isinstance(chunk, dict) and "choices" in chunk:
            token = chunk["choices"][0]["delta"]["content"]
        else:
            token = str(chunk)
        full_response += token

        if '<think>' in token:
            index = token.find('<think>')
            in_think = True
            # Capture text after the <think> tag
            token_after = token[index + len('<think>'):]
            thinking_content += token_after
        elif '</think>' in token:
            # Capture any text before the closing tag and then reset the flag.
            index = token.find('</think>')
            if in_think:
                thinking_content += token[:index]
            in_think = False
        elif in_think:
            # If already within a think block, keep accumulating tokens.
            thinking_content += token

        thinking_placeholder.markdown(
            f'<details class="thinking" open><summary> Model Thinking</summary><pre>{thinking_content}</pre></details>',
            unsafe_allow_html=True
        )
    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned


################### DataFrameSummary Tool #################################

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for LLM Based on dataframe."""
    prompt = f"""
        DataFrame contains {df.shape[0]} rows and {df.shape[1]} columns.
        Data Types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}  # FIXED: Explicitly call
        Columns: {'. '.join(df.columns)}
        Column types: {df.dtypes.to_dict()}

        Provide:
        1. A brief description of what this dataset contains.
        2. 3-4 possible data analysis questions that could be explored.
        3. Keep it concise and focused. 
        """
    return prompt


################### DataInsight Agent #####################################

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses LLM to generate a brief summary and possible questions for the uploaded dataset. """
    prompt = DataFrameSummaryTool(df)
    try:
        messages = [
            {"role": "system", "content": "Detailed thinking on. You are an insightful senior data analyst providing focused insight and briefs."},
            {"role": "user", "content": prompt}
        ]
        # response = 
        # Increase max_tokens to generate full code (adjust as needed)
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,
            max_tokens=512
        )
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating dataset insights : {e}"



############### Helper Function ######################
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



############################### Main Streamlit App ########################################


def main():
    st.set_page_config(layout="wide")
    if "plots" not in st.session_state:
        st.session_state.plots = []

    left, right = st.columns([3,7])

    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253B-v1</a></medium>", unsafe_allow_html=True)
        file = st.file_uploader("Choose CSV", type=["csv"])
        if file:
            if ("df" not in st.session_state) or (st.session_state.get("current_file") != file.name):
                st.session_state.df = pd.read_csv(file)
                st.session_state.current_file = file.name
                st.session_state.messages = []
                with st.spinner("Generating dataset insights …"):
                    st.session_state.insights = DataInsightAgent(st.session_state.df)
            st.dataframe(st.session_state.df.head())
            st.markdown("### Dataset Insights")
            st.markdown(st.session_state.insights)
        else:
            st.info("Upload a CSV to begin chatting with your data.")

    with right:
        st.header("Chat with your data")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Display plot at fixed size
                            st.pyplot(st.session_state.plots[idx], use_container_width=False)

        if file:  # only allow chat after upload
            if user_q := st.chat_input("Ask about your data…"):
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.spinner("Working …"):
                    code, should_plot_flag, code_thinking = CodeGenerationAgent(user_q, st.session_state.df)
                    result_obj = ExecutionAgent(code, st.session_state.df, should_plot_flag)
                    raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                    reasoning_txt = reasoning_txt.replace("`", "")

                # Build assistant response
                is_plot = isinstance(result_obj, (plt.Figure, plt.Axes))
                plot_idx = None
                if is_plot:
                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                    st.session_state.plots.append(fig)
                    plot_idx = len(st.session_state.plots) - 1
                    header = "Here is the visualization you requested:"
                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    header = f"Result: {len(result_obj)} rows" if isinstance(result_obj, pd.DataFrame) else "Result series"
                else:
                    header = f"Result: {result_obj}"

                # Show only reasoning thinking in Model Thinking (collapsed by default)
                thinking_html = ""
                if raw_thinking:
                    thinking_html = (
                        '<details class="thinking">'
                        '<summary>🧠 Reasoning</summary>'
                        f'<pre>{raw_thinking}</pre>'
                        '</details>'
                    )

                # Show model explanation directly 
                explanation_html = reasoning_txt

                # Code accordion with proper HTML <pre><code> syntax highlighting
                code_html = (
                    '<details class="code">'
                    '<summary>View code</summary>'
                    '<pre><code class="language-python">'
                    f'{code}'
                    '</code></pre>'
                    '</details>'
                )
                # Combine thinking, explanation, and code accordion
                assistant_msg = f"{thinking_html}{explanation_html}\n\n{code_html}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_msg,
                    "plot_index": plot_idx
                })
                st.rerun()

if __name__ == "__main__":
    main()
        


# ############### Main Example ######################
# if __name__ == "__main__":
#     # Sample DataFrame for demonstration
#     data = {
#         "Region": ["North", "South", "East", "West"],
#         "Sales": [150, 200, 250, 100]
#     }
#     df = pd.DataFrame(data)
    
#     # Sample query which requests a plot (visualization)
#     query = "Plot the sales by region."
    
#     # 1. Generate code based on query and DataFrame columns.
#     generated_code, should_plot, _ = CodeGenerationAgent(query, df)
#     print("Generated Code:")
#     print(generated_code)
    
#     # 2. Execute the generated code.
#     result = ExecutionAgent(generated_code, df, should_plot)
    
#     # 3. Display the result.
#     # If it's a plot, you can use plt.show() or st.pyplot() if running under Streamlit.
#     if should_plot and isinstance(result, Figure):
#         # Save the figure to a temporary buffer and display
#         buf = io.BytesIO()
#         result.savefig(buf, format='png')
#         buf.seek(0)
#         # For command-line testing, you might save to a file or use result.show()
#         result.show()  # In a local environment, this opens the plot window.
#     else:
#         print("Result:", result)
    
#     # 4. Get the LLM's reasoning on the result.
#     thinking, explanation = ReasoningAgent(query, result)
#     print("\nModel Thinking:")
#     print(thinking)
#     print("\nFinal Explanation:")
#     print(explanation)
