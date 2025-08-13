
# Baltimore Salary Analysis Agent 📈

This project provides an AI-powered agent that can query, analyze, and visualize data from the **Baltimore City Employee Salaries (FY2013) dataset**. The agent understands natural language commands, enabling users to easily explore the data without writing any code.

It is built using OpenAI's `gpt-4o-mini` model and features a local memory system to maintain conversational context. The entire agent is instrumented with **OpenInference** and **Arize Phoenix** for complete observability and tracing of its operations.

-----

## \#\# Features

  * **🔍 Natural Language Querying**: Translates plain English questions into `SQL` queries to fetch data from the local dataset.
  * **🧠 AI-Powered Analysis**: Leverages an LLM to interpret retrieved data, identify trends, and provide insightful summaries.
  * **📊 Dynamic Visualizations**: Generates `Python` code on the fly to create charts and graphs (e.g., bar charts) using `seaborn` and `matplotlib`.
  * **💾 Conversational Memory**:
      * **Buffer Memory**: Remembers the most recent turns of the conversation for immediate context.
      * **Semantic Memory**: Uses a `DuckDB` vector store to retrieve semantically relevant information from past interactions, providing deeper and more accurate context for new queries.
  * **🛰️ Observability**: Integrates with Arize Phoenix for end-to-end tracing of the agent's execution, including LLM calls, tool usage, and data transformations.

-----

## \#\# How It Works

The agent's architecture consists of a central router and a set of specialized tools.

1.  **Router**: The main `run_agent` function acts as a router. It receives the user's prompt, augments it with relevant information from its memory, and then makes a call to the OpenAI model.
2.  **Function Calling**: The agent uses OpenAI's function calling capability to determine which tool is best suited to handle the user's request.
3.  **Tools**: The agent has access to three primary tools:
      * `lookup_salary_data`: Connects to a local CSV file using `DuckDB`, generates an SQL query from the user's prompt, and returns the data as a string.
      * `analyze_salary_data`: Takes the data from the lookup tool and uses the LLM to generate a textual analysis.
      * `generate_visualization`: Takes the data from the lookup tool and a user goal to generate and execute Python code for a chart.
4.  **Execution Loop**: The agent continues to call tools as needed until it has gathered enough information to formulate a final, comprehensive answer for the user.

-----

## \#\# Setup and Installation

Follow these steps to get the agent running on your local machine.

### \#\#\# 1. Prerequisites

  * Python 3.8 or higher.
  * An OpenAI API Key.
  * (Optional) An Arize account for observability.

### \#\#\# 2. Clone the Repository

Clone this project to your local machine.

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### \#\#\# 3. Download the Dataset

Download the dataset from the official Baltimore data portal:

  * **Dataset**: [Baltimore City Employee Salaries FY2013](https://www.google.com/search?q=https://data.baltimorecity.gov/datasets/baltimore::baltimore-city-employee-salaries-fy2013/about)
  * Save the file as `Baltimore_City_Employee_Salaries_FY2013.csv` in your project directory.

### \#\#\# 4. Install Dependencies

Install the required Python libraries using pip.

```bash
pip install openai pandas duckdb pydantic numpy phoenix openinference-instrumentation-openai opentelemetry-sdk matplotlib seaborn
```

### \#\#\# 5. Configure the Script

Open the `Baltimore_with_local_memory.py` file and update the following variables with your credentials and file path:

```python
# Set your OpenAI API Key
openai.api_key = "Your_OpenAI_API_Key"

# (Optional) Set your Arize Phoenix headers and endpoint for tracing
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "Your_OTLP_Headers"
os.environ["PHOENIX_CLIENT_HEADERS"] = "Your_Phoenix_Client_Headers"

# Update with the absolute path to your downloaded dataset
SALARY_DATA_FILE_PATH = '/path/to/your/Baltimore_City_Employee_Salaries_FY2013.csv'
```

-----

## \#\# Usage

To run the agent, execute the script from your terminal:

```bash
python Baltimore_with_local_memory.py
```

The script is pre-configured to run an example query. To ask your own questions, modify the `content` field within the `start_main_span` function call at the bottom of the file:

```python
# Example invocation (will now use memory):
result = start_main_span([{"role": "user",
  "content": "Your question here. For example: 'What are the 10 lowest paying jobs in the dataset?'"}])
print(result)
```

### \#\#\# Example Prompts

Here are a few examples of questions you can ask the agent:


  * `"Tell me the average Gross pay by top 10 job titles where each job has at least 5 employees."`
  * `"Determine which job titles have the highest average Gross value and plot them as a horizontal bar graph."`
  * `"Analyze the salaries for the 'Police Officer' job title."`
  * (As a follow-up): `"Now, create a chart for that data."`

You can test the memory part as below: 

1st Query: "Determine which job titles have the highest average Gross value in the given dataset and provide top 10 ranks overview and plot them as horizontal bar graph using seaborn." 

2nd Query: "Which Job title has second highest average gross?" //This query retrieves data from local memory instead of tool calling again


### \#\#\# Memory File

The script will automatically create a file named `agent_memory.jsonl` in the same directory. This file stores the buffer memory of the conversation and is used to provide context for subsequent queries.
  ```


