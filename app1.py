from dotenv import load_dotenv
from typing import List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from langchain_groq import ChatGroq

from typing import List, Tuple
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (e.g. API keys) from .env file
load_dotenv()

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate summary prompt string for LLM based on dataframe."""
    prompt = f"""
            Dataframe contains {df.shape[0]} rows and {df.shape[1]} columns.
            DataTypes: {df.dtypes.to_dict()}
            Missing values: {df.isnull().sum().to_dict()}
            Columns: {'. '.join(df.columns)}
            Column Types: {df.dtypes.to_dict()}

            Provide:
            1. Brief Description of what this dataset contains.
            2. 4-5 possible data analysis questions that could be explored.
            3. Keep it concise and focused.
            """
    return prompt

def DataInsightAgent(df: pd.DataFrame) -> str:
    """Uses LLM to generate a brief summary and possible questions for uploaded dataset."""
    prompt = DataFrameSummaryTool(df)
    try:
        messages = [
            {"role": "system", "content": "Do detailed thinking. You are a senior data analyst providing focused insight and briefs."},
            {"role": "user", "content": prompt}
        ]
        llm = ChatGroq(
            model = "meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,
            max_tokens=512
        )

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating dataset insights: {e}"
    

# ------------------ Query Understanding Tool ------------------
def QueryUnderstandingTool(query: str) -> bool:
    """Returns True if query requests data visualization based on keywords or LLM fallback."""
    # FAST PATH: simple keyword check (NEW)
    keywords = ['plot', 'chart', 'graph', 'visual', 'histogram', 'scatter', 'line']  # NEW
    if any(kw in query.lower() for kw in keywords):  # NEW
        logger.debug("Detected visualization keyword in query: %s", query)  # NEW
        return True  # NEW

    # FALLBACK: LLM-based detection (CHANGED)
    try:
        messages = [
            {"role": "system", "content": (
                "You are an assistant that determines if a query requests data visualization. "
                "Respond only 'True' or 'False'."
            )},
            {"role": "user", "content": query}
        ]

        llm = ChatGroq(
            model = "meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,
            max_tokens=512
        )

        response = llm.invoke(messages)  # assume llm defined globally or injected
        intent = response.content.strip().lower()
        return intent.startswith("true")  # more robust comparison (CHANGED)
    except Exception as e:
        logger.error("LLM intent detection failed: %s", e)
        return False  # fallback

# ------------------ Prompt Builders ------------------
def PlotGeneratorTool(cols: List[str], query: str) -> str:
    """Generate prompt for LLM to write pandas and matplotlib code for a plot."""
    # Removed seaborn mention (CHANGED)
    return f"""
Given a pandas dataframe 'df' with columns: {', '.join(cols)}.
Write Python code using pandas and matplotlib to answer the query: \"{query}\".

Rules:
1. Import matplotlib.pyplot as plt.
2. Create only one relevant plot with figsize=(6,4) and add appropriate title/labels.
3. Assign final result (DataFrame, Series, scalar or plot figure) to a variable named 'result'.
4. Return only a properly closed ```python code block with no explanations.
"""

def DataAnalysisTool(cols: List[str], query: str) -> str:
    """Generate prompt for LLM to write pandas-only code without plotting."""
    return f"""
Given a pandas dataframe 'df' with columns: {', '.join(cols)}.
Write Python code using pandas to answer the query: \"{query}\".

Rules:
1. Perform necessary data manipulation only; no plotting libraries.
2. Assign final result (DataFrame, Series or scalar) to 'result'.
3. Return only a properly closed ```python code block with no explanations.
"""

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



# ------------------ Main Agent ------------------

def CodeGenerationAgent(query: str, df, llm, max_tokens: int = 1024) -> Tuple[str, bool]:
    """
    Generates code for data analysis or plotting based on query.
    Returns (code, should_plot).
    """
    should_plot = QueryUnderstandingTool(query)

    # Branch prompt based on intent (CHANGED)
    if should_plot:
        prompt = PlotGeneratorTool(df.columns.tolist(), query)
    else:
        prompt = DataAnalysisTool(df.columns.tolist(), query)

    # System message (unchanged)
    system_message = (
        "You are a Python data-analysis expert who writes clean, efficient code. "
        "Solve the given problem with optimal pandas operations. Be concise and focused. "
        "Your response must contain ONLY a properly closed ```python code block with no explanations."
    )
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages, max_tokens=max_tokens)  # CHANGED: allow dynamic tokens
        code = extract_first_code_block(response.content)
        return code, should_plot  # simplified return signature (CHANGED)
    except Exception as e:
        logger.error("Code generation failed: %s", e)
        return "", should_plot  # return empty code on failure

