# Tracing Agent

A Python‐based autonomous data‐analysis agent that uses OpenAI function‐calling, semantic memory, and OTEL tracing to explore the Baltimore City Employee Salaries Dataset (FY2013). The agent can look up data via SQL, perform AI‐powered analysis, and generate visualizations, all while recording spans to Phoenix.

---

## 🚀 Features

- **OpenAI Function Calling** router to dispatch between tools  
- **Tool 1: Database Lookup**  
  - Loads CSV into DuckDB  
  - Auto-generates SQL via LLM  
- **Tool 2: Data Analysis**  
  - AI‐powered summary and trend detection  
- **Tool 3: Data Visualization**  
  - Generates chart configurations and Python plotting code  
- **Semantic Memory**  
  - Embeddings stored in DuckDB  
  - Retrieves relevant past turns via cosine similarity  
- **Observability & Tracing**  
  - Phoenix OTLP collector (OTEL) spans around each call  
- **Retry Logic** for resilient OpenAI calls

---

## 📋 Prerequisites

- Python 3.9+  
- A valid OpenAI API key  
- A Phoenix/OTLP endpoint & headers  
- (Optional) Jupyter notebook if you want inline Markdown output  

---

## 🔧 Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-org/tracing-agent.git
   cd tracing-agent

```shell
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install openai pandas duckdb pydantic phoenix openinference opentelemetry-api

export OPENAI_API_KEY="your_openai_api_key"
export OTEL_EXPORTER_OTLP_HEADERS="your_otlp_headers"
export PHOENIX_CLIENT_HEADERS="your_phoenix_client_headers"
export PHOENIX_COLLECTOR_ENDPOINT="https://app.phoenix.arize.com"

SALARY_DATA_FILE_PATH = "/path/to/Baltimore_City_Employee_Salaries_FY2013.csv"

python Baltimore_with_local_memory.py
```

You can test the memory part with below: 

1st Query: "Determine which job titles have the highest average Gross value in the given dataset and provide top 10 ranks overview and plot them as horizontal bar graph using seaborn." 

2nd Query: "Which Job title has second highest average gross?" //This query retrieves data from local memory instead of tool calling again
