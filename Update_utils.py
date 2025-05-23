import pandas as pd
from langchain.llms import OpenAI
from langchain.llms.base import LlamaLLM

def analyze_code(application_name, cwe_id_name, file_name, line_number, code, code_context=""):
    """
    Constructs a prompt for vulnerability analysis
    """
    # Determine language from file extension
    extension = file_name.split('.')[-1] if '.' in file_name else 'unknown'
    language_map = {
        'ts': 'TypeScript',
        'js': 'JavaScript',
        'py': 'Python',
        'java': 'Java',
        'cs': 'C#',
        'php': 'PHP',
        # Add more mappings as needed
    }
    language = language_map.get(extension, 'Unknown')
    
    prompt = f"""
    As a security analyst, analyze this {language} code for the vulnerability {cwe_id_name}.
    Application: {application_name}
    File: {file_name}
    Line number with potential issue: {line_number}
    
    CODE:
    {code}
    
    {code_context}
    
    Focus on line {line_number} and determine if this is a true {cwe_id_name} vulnerability or a false positive.
    If it's a vulnerability, provide specific mitigation steps.
    
    Is this a false positive? Answer with 'false positive' if it is not a real vulnerability.
    If it is a real vulnerability, provide detailed mitigation steps.
    """
    
    return prompt

def main():
    # Initialize LLM
    llm = LlamaLLM(llm=OpenAI())
    
    # Read CSV file
    data = pd.read_csv('vulnerabilities.csv')
    
    # Add prediction and mitigation columns if they don't exist
    if 'Prediction' not in data.columns:
        data['Prediction'] = ""
    if 'Mitigation' not in data.columns:
        data['Mitigation'] = ""
    
    # Process each row
    for index, row in data.iterrows():
        print(f"Processing {index+1}/{len(data)}: {row['ApplicationName']} - {row['CWE id and name']}")
        
        # Extract file name and line number from Source
        source_parts = row['Source'].split(':')
        file_name = source_parts[0]
        line_number = source_parts[1] if len(source_parts) > 1 else "1"
        
        # Set default code context
        code_context = "File not found. Unable to analyze the code."
        
        # Generate prompt
        prompt = analyze_code(
            row['ApplicationName'], 
            row['CWE id and name'], 
            file_name, 
            line_number, 
            row['Code'],
            code_context
        )
        
        # Get LLM response
        gpt_response = llm._call(prompt)
        
        # Process response
        if "false positive" in gpt_response.lower():
            data.at[index, 'Prediction'] = "FP"
        else:
            data.at[index, 'Prediction'] = "Vulnerability"
            data.at[index, 'Mitigation'] = gpt_response
    
    # Export to Excel
    data.to_excel("Code_Snippet_vulns_Analysis.xlsx", index=False)
    print("Analysis complete. Results saved to Code_Snippet_vulns_Analysis.xlsx")

if __name__ == "__main__":
    main()






# 1. Update your system message function
# Install required libraries if not already installed
# Install required library
# Install required libraries
!pip install requests

import requests
import json
import base64

# Jira connection parameters
server = "https://itrack.web.att.com"
username = dbutils.secrets.get(scope="jira_credentials", key="username")
password = dbutils.secrets.get(scope="jira_credentials", key="password")

# Create basic auth header
auth_str = f"{username}:{password}"
auth_bytes = auth_str.encode('ascii')
base64_bytes = base64.b64encode(auth_bytes)
base64_auth = base64_bytes.decode('ascii')

headers = {
    "Authorization": f"Basic {base64_auth}",
    "Content-Type": "application/json"
}

# Get a single issue to examine its structure
url = f"{server}/rest/api/2/search"
params = {
    "jql": "ORDER BY key ASC",
    "maxResults": 1
}

try:
    # Make the API request
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'issues' in data and len(data['issues']) > 0:
            issue = data['issues'][0]
            
            # Print the issue key
            print(f"Examining issue: {issue['key']}")
            
            # Print all available fields
            print("\nAll available fields:")
            for field_name, field_value in issue['fields'].items():
                field_type = type(field_value).__name__
                
                # For complex objects, show a sample of their structure
                if isinstance(field_value, dict):
                    print(f"  {field_name} (dict): {json.dumps(field_value)[:100]}...")
                elif isinstance(field_value, list):
                    if field_value:
                        print(f"  {field_name} (list): {json.dumps(field_value)[:100]}...")
                    else:
                        print(f"  {field_name} (empty list)")
                else:
                    print(f"  {field_name} ({field_type}): {str(field_value)[:100]}")
            
            # Look specifically for status field
            if 'status' in issue['fields']:
                print("\nStatus field structure:")
                print(json.dumps(issue['fields']['status'], indent=2))
            
            # Look for fields that might contain "team" or "scrum"
            print("\nPossible Scrum Team fields:")
            for field_name, field_value in issue['fields'].items():
                if field_name.lower().startswith('customfield_'):
                    field_str = str(field_value).lower()
                    if field_value and ('team' in field_str or 'scrum' in field_str):
                        print(f"  {field_name}: {field_value}")
            
            # Look for fix versions
            if 'fixVersions' in issue['fields']:
                print("\nFix Versions field structure:")
                print(json.dumps(issue['fields']['fixVersions'], indent=2))
        else:
            print("No issues found")
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"Error: {str(e)}")




!pip install jira

from jira import JIRA
import pandas as pd
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.appName("JiraDataExtraction").getOrCreate()

# Jira connection parameters
server = "https://itrack.web.att.com"
username = dbutils.secrets.get(scope="jira_credentials", key="username")
password = dbutils.secrets.get(scope="jira_credentials", key="password")

# Connect to Jira
jira = JIRA(server=server, basic_auth=(username, password))

# Define the fields we want to fetch
fields = ["status", "fixVersions", "customfield_10123"]  # Add the custom field ID for Scrum Team

# Function to fetch all issues with pagination, only requesting specific fields
def fetch_all_issues(batch_size=1000):
    all_data = []
    start_at = 0

    print("Starting to fetch Jira tickets...")

    while True:
        # Fetch a batch of issues with only the fields we need
        print(f"Fetching batch starting at position {start_at}...")
        batch = jira.search_issues("ORDER BY key ASC", startAt=start_at, maxResults=batch_size, fields=fields)

        # If no more issues, break the loop
        if len(batch) == 0:
            break

        # Process each issue in the batch
        for issue in batch:
            # Get Key
            key = issue.key

            # Get Status
            status = issue.fields.status.name

            # Get Fix Versions
            fix_versions = []
            for version in issue.fields.fixVersions:
                fix_versions.append(version.name)
            fix_versions_str = ", ".join(fix_versions) if fix_versions else ""

            # Get Scrum Team (custom field)
            scrum_team = ""
            team_field = getattr(issue.fields, 'customfield_10123', None)
            if team_field and hasattr(team_field, 'value'):
                scrum_team = team_field.value

            # Add to data collection
            all_data.append({
                "Key": key,
                "Status": status,
                "Fix_Versions": fix_versions_str,
                "Scrum_Team": scrum_team
            })

        # Update start position for next batch
        start_at += len(batch)
        print(f"Processed {start_at} issues so far")

        # If batch is smaller than requested size, we've reached the end
        if len(batch) < batch_size:
            break

    print(f"Completed fetching {len(all_data)} issues")
    return all_data

# Fetch all issues and process them directly
data = fetch_all_issues()

# Create DataFrame
df = pd.DataFrame(data)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Save to Delta table
spark_df.write.format("delta").mode("overwrite").saveAsTable("jira_tickets")

# Save to CSV
spark_df.toPandas().to_csv("/dbfs/FileStore/jira_tickets.csv", index=False)

# Display sample
print("Sample of fetched data:")
display(spark_df.limit(10))

print(f"Total tickets: {len(data)}")
print("Data saved to Delta table 'jira_tickets' and CSV file '/dbfs/FileStore/jira_tickets.csv'")





!pip install jira

from jira import JIRA
import pandas as pd
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("JiraDataExtraction").getOrCreate()

# Jira connection parameters
server = "https://itrack.web.att.com"
# For security, use Databricks secrets instead of hardcoding credentials
username = dbutils.secrets.get(scope="jira_credentials", key="username")
password = dbutils.secrets.get(scope="jira_credentials", key="password")

# Connect to Jira
jira = JIRA(server=server, basic_auth=(username, password))

# JQL query to get all tickets
# You can adjust this query as needed
jql_query = "ORDER BY key DESC"

# Function to fetch all issues with pagination
def fetch_all_issues(jql, batch_size=1000):
    issues = []
    start_at = 0
    
    while True:
        batch = jira.search_issues(jql, startAt=start_at, maxResults=batch_size)
        if len(batch) == 0:
            break
            
        issues.extend(batch)
        start_at += len(batch)
        print(f"Fetched {start_at} issues so far...")
        
        # If batch is smaller than batch_size, we've reached the end
        if len(batch) < batch_size:
            break
            
    return issues

# Fetch all issues
print("Fetching all Jira tickets, this may take some time...")
all_issues = fetch_all_issues(jql_query)
print(f"Total tickets fetched: {len(all_issues)}")

# Extract required fields from each issue
data = []
for issue in all_issues:
    # Extract fields
    key = issue.key
    status = issue.fields.status.name
    
    # Handle fix versions (may be multiple or none)
    fix_versions = []
    for version in issue.fields.fixVersions:
        fix_versions.append(f"{version.name} ({version.releaseDate})" if hasattr(version, 'releaseDate') else version.name)
    fix_versions_str = " - ".join(fix_versions) if fix_versions else ""
    
    # Get Scrum Team (custom field - adjust field ID as needed)
    # You may need to inspect your Jira instance to get the correct field ID
    scrum_team = getattr(issue.fields, 'customfield_10123', None)
    if scrum_team and hasattr(scrum_team, 'value'):
        scrum_team = scrum_team.value
    
    # Add more fields as needed
    summary = issue.fields.summary
    issue_type = issue.fields.issuetype.name
    project = issue.fields.project.key
    
    data.append({
        "Key": key,
        "Summary": summary,
        "Status": status,
        "Issue_Type": issue_type,
        "Project": project,
        "Fix_Versions": fix_versions_str,
        "Scrum_Team": scrum_team
        # Add more fields as needed
    })

# Create DataFrame
df = pd.DataFrame(data)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Save to Delta table or other format
spark_df.write.format("delta").mode("overwrite").saveAsTable("all_jira_tickets")

# Display sample data
print("Sample of fetched tickets:")
display(spark_df.limit(10))

# Optionally export to CSV
spark_df.toPandas().to_csv("/dbfs/FileStore/all_jira_tickets.csv", index=False)
print("Data saved to CSV at /dbfs/FileStore/all_jira_tickets.csv")

def sanitize_json_string(json_str):
    # First, let's try to identify if we have a JSON-like structure
    if not (json_str.strip().startswith('{') and json_str.strip().endswith('}')):
        print("Input doesn't appear to be JSON format")
        return '{}'
    
    try:
        # Try direct parsing first
        return json_str
    except:
        pass
    
    # Manual JSON field extraction and cleaning
    cleaned_json = {}
    
    # Extract each field individually using regex
    import re
    
    # Extract predicted_label
    label_match = re.search(r'"predicted_label"\s*:\s*"([^"]*)"', json_str)
    if label_match:
        cleaned_json["predicted_label"] = label_match.group(1)
    else:
        cleaned_json["predicted_label"] = "unknown"
    
    # Extract emotion_summary
    emotion_match = re.search(r'"emotion_summary"\s*:\s*"([^"]*)"', json_str)
    if emotion_match:
        cleaned_json["emotion_summary"] = emotion_match.group(1)
    else:
        cleaned_json["emotion_summary"] = "Neutral"
    
    # Extract emotion_confidence
    confidence_match = re.search(r'"emotion_confidence"\s*:\s*([\d\.]+)', json_str)
    if confidence_match:
        cleaned_json["emotion_confidence"] = float(confidence_match.group(1))
    else:
        cleaned_json["emotion_confidence"] = 0.5
    
    # Extract rationale (this is trickier due to quotes)
    rationale_start = json_str.find('"rationale"')
    if rationale_start != -1:
        # Find the colon after "rationale"
        colon_pos = json_str.find(':', rationale_start)
        if colon_pos != -1:
            # Find the first quote after the colon
            first_quote = json_str.find('"', colon_pos)
            if first_quote != -1:
                # Find the closing quote (this is tricky with nested quotes)
                # Look for a quote followed by a comma or closing brace
                rationale_end = -1
                for i in range(first_quote + 1, len(json_str)):
                    if json_str[i] == '"' and (i+1 < len(json_str) and (json_str[i+1] == ',' or json_str[i+1] == '}')):
                        rationale_end = i
                        break
                
                if rationale_end != -1:
                    # Extract the rationale text
                    rationale_text = json_str[first_quote+1:rationale_end]
                    # Clean it up (remove problematic characters)
                    rationale_text = rationale_text.replace('"', "'").replace('\\', '')
                    cleaned_json["rationale"] = rationale_text
    
    if "rationale" not in cleaned_json:
        cleaned_json["rationale"] = "No rationale provided"
    
    # Extract affected_section
    section_match = re.search(r'"affected_section"\s*:\s*"([^"]*)"', json_str)
    if section_match:
        cleaned_json["affected_section"] = section_match.group(1)
    else:
        cleaned_json["affected_section"] = "unknown section"
    
    # Extract improvement_suggestion (similar to rationale)
    improvement_start = json_str.find('"improvement_suggestion"')
    if improvement_start != -1:
        colon_pos = json_str.find(':', improvement_start)
        if colon_pos != -1:
            first_quote = json_str.find('"', colon_pos)
            if first_quote != -1:
                improvement_end = -1
                for i in range(first_quote + 1, len(json_str)):
                    if json_str[i] == '"' and (i+1 < len(json_str) and (json_str[i+1] == ',' or json_str[i+1] == '}')):
                        improvement_end = i
                        break
                
                if improvement_end != -1:
                    improvement_text = json_str[first_quote+1:improvement_end]
                    improvement_text = improvement_text.replace('"', "'").replace('\\', '')
                    cleaned_json["improvement_suggestion"] = improvement_text
    
    if "improvement_suggestion" not in cleaned_json:
        cleaned_json["improvement_suggestion"] = "No improvement suggestion provided"
    
    # Convert back to JSON string
    import json
    return json.dumps(cleaned_json)




def get_system_message():
    delimiter = "####"
    
    labels_str = ", ".join(prediction_labels)
    emotions_str = ", ".join(emotion_labels)
    
    system_message = f"""You are an advanced AI assistant analyzing user queries and user queries will be delimited with {delimiter} characters.
Analyze properly and classify each query into ONLY one of these specific categories:
{labels_str}. If you cannot classify the query, return "unknown".

Also determine the emotional tone of the comment from ONLY these emotions:
{emotions_str}. Do not create new emotion categories.

Additionally, provide:
1. A detailed rationale explaining why you classified the comment this way
2. Which section of the service the issue relates to
3. What specific improvements could address the issue

Return with a JSON object containing:
{{
    "predicted_label": one of the specified labels listed above or "unknown",
    "emotion_summary": one of the specified emotions listed above,
    "emotion_confidence": confidence score between 0 and 1,
    "rationale": detailed explanation of why this classification was chosen,
    "affected_section": specific section or feature where the issue occurred,
    "improvement_suggestion": actionable recommendation to address the issue
}}

IMPORTANT: Do not include the delimiter '{delimiter}' in your response. Return only the JSON object.
IMPORTANT: Only use the exact emotions and labels provided in the lists above.
"""
    return system_message

# 2. Modify your classify_comment function to extract the new fields
def classify_comment(user_comment):
    system_message = get_system_message()
    messages = build_prompt_message(system_message, user_comment)
    
    try:
        # Your existing LLaMA model call
        completion = llm.call(messages)
        response_json = json.loads(completion)
        
        # Extract all fields including the new ones
        prediction = response_json.get("predicted_label", "unknown")
        emotion = response_json.get("emotion_summary", "Neutral")
        confidence = response_json.get("emotion_confidence", 0.5)
        rationale = response_json.get("rationale", "No rationale provided")
        section = response_json.get("affected_section", "Unknown section")
        improvement = response_json.get("improvement_suggestion", "No improvement suggestion provided")
        
        # Return tuple with all fields
        return (prediction, emotion, confidence, rationale, section, improvement)
    except Exception as e:
        print(f"Error in LLM classification: {str(e)}")
        return ("unknown", "Neutral", 0.5, "Error occurred during classification", "N/A", "Check system logs for errors")

# 3. Update your process_chunk function to handle the new return values
def process_chunk(chunk_df):
    chunk_results = {}
    
    for idx, text in chunk_df.items():
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            chunk_results[idx] = (None, None, None, None, None, None)
        else:
            # Call the enhanced classification function
            result = classify_comment(str(text))
            chunk_results[idx] = result
    
    return chunk_results


def sanitize_json_string(json_str):
    # Replace escaped quotes with temporary placeholder
    temp_str = json_str.replace('\\"', '___QUOTE___')
    # Replace nested quotes in values with single quotes
    temp_str = re.sub(r'": "([^"]*)"([^"]*)"([^"]*)"', r'": "\1\'\2\'\3"', temp_str)
    # Restore escaped quotes
    return temp_str.replace('___QUOTE___', '\\"')

try:
    sanitized_completion = sanitize_json_string(completion)
    response_json = json.loads(sanitized_completion)
    # Rest of your code...


# 4. Add the new columns to your results dataframe in your main processing function
result_columns = [
    "predicted_label",
    "emotion_summary", 
    "emotion_confidence",
    "rationale",
    "affected_section",
    "improvement_suggestion"
]




def sanitize_json_string(json_str):
    try:
        # First attempt to parse as is - might already be valid
        json.loads(json_str)
        return json_str
    except:
        # If parsing fails, try to fix common issues
        pass
    
    # Handle common JSON issues
    try:
        # Replace problematic escape sequences
        temp_str = json_str
        
        # Fix common escape sequence issues
        temp_str = temp_str.replace('\\', '\\\\')  # Double all backslashes first
        temp_str = temp_str.replace('\\\\\"', '\\"')  # Fix double escaped quotes
        temp_str = temp_str.replace('\\\\n', '\\n')  # Fix double escaped newlines
        
        # Remove any control characters
        temp_str = ''.join(ch for ch in temp_str if ord(ch) >= 32 or ch in '\n\r\t')
        
        # Try to parse the sanitized string
        json.loads(temp_str)
        return temp_str
    except:
        # If still failing, try a more aggressive approach
        pass
    
    # More aggressive sanitization
    try:
        # Extract what looks like JSON using regex
        import re
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, json_str, re.DOTALL)
        if match:
            potential_json = match.group(1)
            # Try to parse
            json.loads(potential_json)
            return potential_json
    except:
        pass
    
    # If all else fails, return a valid but empty JSON
    return '{}'

