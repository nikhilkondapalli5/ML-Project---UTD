#!/usr/bin/env python
# coding: utf-8

#Building the Agent 

# The agent is comprised of a router using OpenAI function calling, and a set of three tools: a database lookup tool, a data analysis tool, and a code generator to create graphs.

# The agent can lookup information from a local file, perform analysis on that information, and graph results. The example local file is the data of Baltimore City Employee Salaries Dataset (FY2013) . The agent can help the Govt. Organisations understand trends and anomalies in their salaries data.
# ## Importing necessary libraries 

from openai import OpenAI
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
import logging
import numpy as np
from functools import wraps
from IPython.display import Markdown
import openai
import warnings
warnings.filterwarnings('ignore')
import phoenix as px
import os
import numpy as np
import time
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from openinference.instrumentation import TracerProvider



# ## Setting up the OpenAI API key and the Phoenix Collector Endpoint
openai.api_key = "Your_OpenAI_API_Key"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Your_OTLP_Headers";
os.environ["PHOENIX_CLIENT_HEADERS"] = "Your_Phoenix_Client_Headers";
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com";

if openai.api_key is None:
    raise ValueError("API key not set. Please set the OPENAI_API_KEY environment variable.")

PROJECT_NAME = "tracing-agent"
tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint= os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
)

OpenAIInstrumentor().instrument(tracer_provider = tracer_provider)
tracer = tracer_provider.get_tracer(__name__)


MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MEMORY_FILE = "agent_memory.jsonl"
MAX_BUFFER_TURNS = 10
MEM_VECTOR_TABLE = "semantic_memory"


# Setup semantic memory vector store (DuckDB)
con = duckdb.connect(database=':memory:')
# Create vector table if not exists
# NEW: double[] column
con.execute(f"CREATE TABLE IF NOT EXISTS {MEM_VECTOR_TABLE} (text TEXT, embedding DOUBLE[])")


# --- Logging ---
warnings = logging.getLogger("py.warnings")
warnings.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Retry Decorator ---
def retry_on_exception(max_retries=3, delay=2, allowed_exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    logger.warning(f"[Retry {attempt}/{max_retries}] {func.__name__} failed: {e}")
                    if attempt == max_retries:
                        raise
                    time.sleep(delay * attempt)
        return wrapper
    return decorator

@retry_on_exception(max_retries=3, delay=1)
def call_openai_with_retry(**kwargs):
    return openai.chat.completions.create(**kwargs)

# --- Semantic Memory Functions ---
def load_buffer_memory() -> list[dict]:
    """Return last MAX_BUFFER_TURNS turns from MEMORY_FILE."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        lines = f.readlines()[-MAX_BUFFER_TURNS:]
    entries = [json.loads(line) for line in lines]
    return [e for e in entries if e['role'] in ('user', 'assistant')]


def save_buffer_turn(role: str, content: str):
    """Append a single turn to MEMORY_FILE."""
    with open(MEMORY_FILE, 'a') as f:
        f.write(json.dumps({'role': role, 'content': content}) + '\n')


def get_embedding(text: str) -> list[float]:
    resp = openai.embeddings.create(input=text, model=EMBED_MODEL)
    return resp.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    return float(a_np.dot(b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def add_semantic_memory(text: str):
    emb = get_embedding(text)  # a Python list of floats
    con.execute(
        f"INSERT INTO {MEM_VECTOR_TABLE} (text, embedding) VALUES (?, ?)",
        [text, emb]
    )


def retrieve_semantic_memory(query: str, top_k: int = 5) -> list[str]:
    q_emb = get_embedding(query)
    df = con.execute(f"SELECT text, embedding FROM {MEM_VECTOR_TABLE}").df()
    df['sim'] = df.embedding.apply(lambda e: cosine_similarity(e, q_emb))
    return df.nlargest(top_k, 'sim')['text'].tolist()



# ## Defining the tools

# Let's start by creating the three tools the agent will be able to use.

# ### Tool 1: Database Lookup

# This first tool reads from a local csv file that contains the salary data. 

# define the path to the salary data
SALARY_DATA_FILE_PATH = '/Users/konda/Downloads/Baltimore_City_Employee_Salaries_FY2013.csv'
# This database lookup tool works using three steps. 

# 1. First, it creates the SQL table from a local file, if not already done.
# 2. Second, it translates the original prompt into an sql query (using an LLM call).
# 3. Finally, it runs that query against the database.

# prompt template for step 2 of tool 1
SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}
The available columns are: {columns}
The table name is: {table_name}
"""

# code for step 2 of tool 1
def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, 
                                                    columns=columns, 
                                                    table_name=table_name)
    response = call_openai_with_retry(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@tracer.tool()  
# code for tool 1sk-proj-PDZjwVMlPMTicHqKTwiI
def lookup_salary_data(prompt: str) -> str:
    """Implementation of salary data lookup from csv file using SQL"""
    try:
        # define the table name
        table_name = "salary"

        # step 1: read the csv file into a DuckDB table
        df = pd.read_csv(SALARY_DATA_FILE_PATH)
        #df.columns = [col.replace("-", "_") for col in df.columns]
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")

        # step 2: generate the SQL code.
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        # clean the response to make sure it only includes the SQL code
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")
        #duckdb.sql(f"DESCRIBE {table_name}").show()

        # step 3: execute the SQL query
        result = duckdb.sql(sql_query).df()
        print(result)
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"

#Testcase:
#example_data = lookup_salary_data("Tell me the average Gross by top 10 job titles in the dataset where each job title has atleast 5 values.")
#print(example_data)

# ### Tool 2: Data Analysis
# The second tool can analyze the returned data and display conclusions to users.
# Construct prompt based on analysis type and data subset
DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

# code for tool 2
@tracer.tool()
def analyze_salary_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered salary data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)
    response = call_openai_with_retry(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    
    analysis = response.choices[0].message.content
    return analysis if analysis else "No analysis could be generated"

#Test:
#print(analyze_salary_data(prompt="what trends do you see in this data", data=example_data))

# ### Tool 3: Data Visualization
# The third tool generates python code to create the requested graphs from the returned data of the first tool. It consists of two steps:
# 1. First, it creates the chart configuration: chart type, title, data, lables for x-axis and y-axis (using an LLM call).
# 2. Second, it generates the python code based on the chart configuration of the first step (using an LLM call).

# prompt template for step 1 of tool 3
CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

# class defining the response format of step 1 of tool 3
class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")

# code for step 1 of tool 3
@tracer.chain()
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    """Generate chart visualization configuration
    
    Args:
        data: String containing the data to visualize
        visualization_goal: Description of what the visualization should show
        
    Returns:
        Dictionary containing line chart configuration
    """
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data,
                                                         visualization_goal=visualization_goal)
    
    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format=VisualizationConfig,
    )
    
    try:
        # Extract axis and title info from response
        content = response.choices[0].message.content
        
        # Return structured chart config
        return {
            "chart_type": content.chart_type,
            "x_axis": content.x_axis,
            "y_axis": content.y_axis,
            "title": content.title,
            "data": data
        }
    except Exception:
        return {
            "chart_type": "line", 
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data
        }

# prompt template for step 2 of tool 3
CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""
@tracer.chain()
# code for step 2 of tool 3
def create_chart(config: dict) -> str:
    """Create a chart based on the configuration"""
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)
    
    response = call_openai_with_retry(
        model=MODEL,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    
    code = response.choices[0].message.content
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()
    
    return code

@tracer.tool()  
# code for tool 3
def generate_visualization(data: str, visualization_goal: str) -> str:
    """Generate a visualization based on the data and goal"""
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    print(code)
    exec(code)
    return code

# Test:
#sample = lookup_salary_data("Tell me the top 5 job titles based on their average Gross where the the job title has atleast 5 values.")
#code = generate_visualization(example_data, "A bar chart of top 5 job titles by their names vs their respective average Gross. Put the Job titles on the x-axis and the average gross on the y-axis.")

# ## Defining the Router
# Now that all of the tools are defined, you can create the router. The router will take the original user input, and is responsible for calling any tools. After each tool call is completed, the agent will return to router to determine whether another tool should be called.
# ### Tool Schema
# Defining the tools in a way that can be understood by our OpenAI model. OpenAI understands a specific JSON format:
# Define tools/functions that can be called by the model

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_salary_data",
            "description": "Look up data from Baltimore City Salaries dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_salary_data", 
            "description": "Analyze salary data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The lookup_salary_data tool's output."},
                    "prompt": {"type": "string", "description": "The unchanged prompt that the user provided."}
                },
                "required": ["data", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object", 
                "properties": {
                    "data": {"type": "string", "description": "The lookup_salary_data tool's output."},
                    "visualization_goal": {"type": "string", "description": "The goal of the visualization."}
                },
                "required": ["data", "visualization_goal"]
            }
        }
    }
]

# Dictionary mapping function names to their implementations
tool_implementations = {
    "lookup_salary_data": lookup_salary_data,
    "analyze_salary_data": analyze_salary_data, 
    "generate_visualization": generate_visualization
}

# ### Router Logic
# The router is composed of a main loop method, and a method to handle the tool calls that you get back from the model.
# The following defines the function `handle_tool_calls` and the variable `SYSTEM_PROMPT`, which will be used by the function `run_agent` defining the router logic.
# code for executing the tools returned in the model's response

@tracer.chain()
def handle_tool_calls(tool_calls, messages):
    for call in tool_calls:
        fn = tool_implementations[call.function.name]
        args = json.loads(call.function.arguments)
        result = fn(**args)
        # append function result
        messages.append({
            'role': 'function',
            'name': call.function.name,
            'content': result
        })
        save_buffer_turn('assistant', result)
        add_semantic_memory(result)
    return messages


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Baltimore City salaries dataset.
"""
def run_agent(raw_messages):
    # Normalize
    if isinstance(raw_messages, str):
        incoming = [{'role': 'user', 'content': raw_messages}]
    else:
        incoming = raw_messages

    # Save and index user messages
    for msg in incoming:
        if msg['role'] == 'user':
            save_buffer_turn('user', msg['content'])
            add_semantic_memory(msg['content'])

    # Build context: include buffer + relevant semantic memory
    buffer = load_buffer_memory()
    recent_memory = retrieve_semantic_memory(incoming[-1]['content'])
    messages = buffer + [{'role': 'system', 'content': f"Semantic memory: {recent_memory}"}] + incoming

    # Ensure system prompt at start
    if not any(m['role'] == 'system' for m in messages):
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

    # Main loop
    while True:
        request_msgs = [m for m in messages if m['content']]
        response = call_openai_with_retry(
            model=MODEL,
            messages=request_msgs,
            tools=tools
        )
        choice = response.choices[0].message

        # Save assistant response
        if choice.content:
            save_buffer_turn('assistant', choice.content)
            add_semantic_memory(choice.content)

        messages.append({'role': choice.role, 'content': choice.content})

        # If function call requested
        if getattr(choice, 'tool_calls', None):
            messages = handle_tool_calls(choice.tool_calls, messages)
            continue

        return choice.content



def start_main_span(messages):
    with tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:
        span.set_input(value=messages)
        result = run_agent(messages)
        span.set_output(value=result)
        span.set_status(StatusCode.OK)
        return result


# Example invocation (will now use memory):
result = start_main_span([{"role": "user",
    "content": "Determine which job titles have the highest average Gross value in the given dataset and provide top 10 ranks overview and plot them as horizontal bar graph using seaborn."}])
print(result)
