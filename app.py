# Enhanced implementation with error handling and memory

# Import necessary libraries
import autogen
from llama import LlamaLLM
import json
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LlamaLLM client
llm_client = LlamaLLM(llm_model="openai")

# Custom LLM handler with error handling
def custom_llm_handler(messages, **kwargs):
    try:
        # Format messages for your LlamaLLM client
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        response = llm_client.generate(prompt=prompt)
        return {"content": response}
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return {"content": f"Error processing request: {str(e)}"}

# Memory class for agents
class AgentMemory:
    def __init__(self):
        self.conversations = []
        self.context = {}
    
    def add_conversation(self, query, response, agent):
        self.conversations.append({
            "query": query,
            "response": response,
            "agent": agent,
            "timestamp": autogen.get_utc_time()
        })
    
    def update_context(self, key, value):
        self.context[key] = value
    
    def get_recent_conversations(self, n=5):
        return self.conversations[-n:] if len(self.conversations) > 0 else []
    
    def get_context(self, key=None):
        if key:
            return self.context.get(key)
        return self.context

# Initialize memory
memory = AgentMemory()

# Autogen config with custom handler
llm_config = {
    "cache_seed": 42,
    "temperature": 0.7,
    "config_list": [{"model": "gpt-4-32k"}],
    "custom_llm_provider": custom_llm_handler
}

# Enhanced system messages with context awareness
intent_classifier = autogen.AssistantAgent(
    name="IntentClassifier",
    llm_config=llm_config,
    system_message="""Analyze user queries and determine if they require:
    1. Snowflake data access (SQL, data warehousing, structured data)
    2. Power BI reporting (dashboards, visualizations, reports)
    Respond with only: "SNOWFLAKE" or "POWERBI" or "BOTH" if both are needed."""
)

snowflake_agent = autogen.AssistantAgent(
    name="SnowflakeAgent",
    llm_config=llm_config,
    system_message="""You handle Snowflake database queries.
    You can generate SQL, explain data structures, and help with data extraction.
    Always verify SQL syntax before execution and handle errors gracefully."""
)

powerbi_agent = autogen.AssistantAgent(
    name="PowerBIAgent",
    llm_config=llm_config,
    system_message="""You handle Power BI reporting queries.
    You can explain DAX, report creation, visualization best practices, and dashboard design.
    Consider user's visualization needs and data context when providing recommendations."""
)

# User proxy with enhanced capabilities
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"}
)

# GroupChat with all agents
groupchat = autogen.GroupChat(
    agents=[user_proxy, intent_classifier, snowflake_agent, powerbi_agent],
    messages=[],
    max_round=12
)
manager = autogen.GroupChatManager(groupchat=groupchat)

# Enhanced Snowflake connection with error handling
def connect_to_snowflake():
    try:
        import snowflake.connector
        # Connection parameters from secure storage
        conn = snowflake.connector.connect(
            user='your_username',
            password='your_password',
            account='your_account',
            warehouse='your_warehouse',
            database='your_database'
        )
        return conn
    except Exception as e:
        logger.error(f"Snowflake connection error: {str(e)}")
        raise

# Enhanced Snowflake query execution
def execute_snowflake_query(query):
    try:
        conn = connect_to_snowflake()
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        results = cursor.fetchall()
        
        # Format results as list of dictionaries
        formatted_results = []
        for row in results:
            formatted_results.append(dict(zip(columns, row)))
        
        # Store query results in memory
        memory.update_context("last_snowflake_query", query)
        memory.update_context("last_snowflake_results", formatted_results)
        
        return formatted_results
    except Exception as e:
        error_msg = f"Error executing Snowflake query: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# Power BI connection and query functions
def connect_to_powerbi():
    try:
        # Implement Power BI connection logic
        # This is a placeholder - implement based on your Power BI setup
        return "PowerBI connection"
    except Exception as e:
        logger.error(f"Power BI connection error: {str(e)}")
        raise

def query_powerbi(dataset_id, report_id=None):
    try:
        # Placeholder for Power BI query implementation
        results = {"dataset": dataset_id, "report": report_id, "data": "Sample data"}
        
        # Store query results in memory
        memory.update_context("last_powerbi_query", {"dataset": dataset_id, "report": report_id})
        memory.update_context("last_powerbi_results", results)
        
        return results
    except Exception as e:
        error_msg = f"Error querying Power BI: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

# Register functions with agents
snowflake_agent.register_function(
    function=execute_snowflake_query,
    name="execute_snowflake_query",
    description="Executes a SQL query on Snowflake and returns results"
)

powerbi_agent.register_function(
    function=query_powerbi,
    name="query_powerbi",
    description="Queries Power BI dataset or report and returns results"
)

# Enhanced query processing with context awareness
def process_user_query(query):
    try:
        # Add previous context if available
        context = ""
        recent_convos = memory.get_recent_conversations(3)
        if recent_convos:
            context = "Previous context:\n" + "\n".join([
                f"Q: {c['query']}\nA: {c['response']}" for c in recent_convos
            ])
            
        query_with_context = f"{context}\n\nCurrent query: {query}" if context else query
        
        # Determine intent
        user_proxy.initiate_chat(intent_classifier, message=query_with_context)
        intent_response = user_proxy.last_message()["content"].strip()
        
        # Route based on intent
        if "BOTH" in intent_response.upper():
            # Use group chat for queries requiring both systems
            user_proxy.initiate_chat(manager, message=query_with_context)
            response = user_proxy.last_message()["content"]
        elif "SNOWFLAKE" in intent_response.upper():
            user_proxy.initiate_chat(snowflake_agent, message=query_with_context)
            response = user_proxy.last_message()["content"]
        elif "POWERBI" in intent_response.upper():
            user_proxy.initiate_chat(powerbi_agent, message=query_with_context)
            response = user_proxy.last_message()["content"]
        else:
            # Default to group chat if intent is unclear
            user_proxy.initiate_chat(manager, message=query_with_context)
            response = user_proxy.last_message()["content"]
        
        # Store in memory
        memory.add_conversation(query, response, intent_response)
        
        return response
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Example usage function
def run_example():
    queries = [
        "What were our top 10 products by revenue last quarter?",
        "Create a visualization showing sales trends by region",
        "Compare our Q1 performance with the forecast in our executive dashboard"
    ]
    
    for query in queries:
        print(f"\n\nQUERY: {query}")
        response = process_user_query(query)
        print(f"RESPONSE: {response}")






# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load your saved multi-output regression model
# model = joblib.load('xgmulti.pickle_new')

# # Create the Streamlit web app
# st.title('Multi-Output Regression Predictor')

# # Create input fields for 16 variables
# input_data = []
# for i in range(16):
#    input_data.append(st.number_input(f'Input {i+1}', value=0.0))

# # Create a button to trigger the prediction
# if st.button('Predict'):
#    # Prepare the input data for the model
#    input_array = np.array(input_data).reshape(1, -1)

#    # Make predictions using the loaded model
#    predictions = model.predict(input_array)

#    # Display the results for the 6 output variables
#    for i, pred in enumerate(predictions[0]):
#        st.write(f'Output {i+1}: {pred}')

# # Add any other UI elements as needed

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your saved multi-output regression model
model = joblib.load('xgmulti.pickle_new')

# Create the Streamlit web app
st.title('Multi-Output Regression Predictor')

# Create a dictionary to map variable names to their input values
variable_names = {
   "Variable 1": st.number_input('LK_RD_', value=0.0),
   "Variable 2": st.number_input('HK_RD_', value=0.0),
   "Variable 3": st.number_input('LGO_DRYER_INLET_FLOW_', value=0.0),
   "Variable 4": st.number_input('HGO_RUNDOWN_', value=0.0),
   "Variable 5": st.number_input('HK_PA_', value=0.0),
   "Variable 6": st.number_input('LGO_PA_', value=0.0),
   "Variable 7": st.number_input('HGO_PA_', value=0.0),
   "Variable 8": st.number_input('LK_STRIPPING_STEAM_', value=0.0),
   "Variable 9": st.number_input('A-COL_STRIPPING_STEAM_', value=0.0),
   "Variable 10": st.number_input('EXP_NAPH', value=0.0),
   "Variable 11": st.number_input('EXP_KERO', value=0.0),
   "Variable 12": st.number_input('EXP_DSL', value=0.0),
   "Variable 13": st.number_input('EXP_RC', value=0.0),
   "Variable 14": st.number_input('Column_overhead_pressure', value=0.0),
   "Variable 15": st.number_input('column_top_temperature', value=0.0),
   "Variable 16": st.number_input('F-101_COMMON_OUT', value=0.0)
   
   # Add more variables as needed
}

# Create a button to trigger the prediction
if st.button('Predict'):
   # Prepare the input data for the model
   input_data = [value for value in variable_names.values()]
   input_array = np.array(input_data).reshape(1, -1)

   # Make predictions using the loaded model
   predictions = model.predict(input_array)

   # Display the results for the 6 output variables
   for i, pred in enumerate(predictions[0]):
       st.write(f'Output {i+1}: {pred}')

# Add any other UI elements as needed
