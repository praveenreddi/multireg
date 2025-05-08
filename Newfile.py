# Create a custom wrapper around the LLM to handle the response properly
class SafeLLMWrapper:
    def __init__(self, llm):
        self.llm = llm
    
    def call(self, prompt):
        try:
            # Get the raw response
            response = self.llm._call(prompt)
            
            # If it's already a string, return it directly
            if isinstance(response, str):
                return {"content": response}
            
            # If it's a dict with content, return it as is
            if isinstance(response, dict) and 'content' in response:
                return response
                
            # Otherwise, convert to a dict with content
            return {"content": str(response)}
            
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            return {"content": f"Error: {str(e)}"}

# In your main code:
llm = LlamaLLM(llm=OpenAI())
safe_llm = SafeLLMWrapper(llm)

# Then use safe_llm.call(prompt) instead of llm._call(prompt)
response_dict = safe_llm.call(prompt)
mitigation_text = response_dict["content"]





import pandas as pd
from langchain.llms import OpenAI
from langchain.llms.base import LlamaLLM

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

        # Construct prompt for LLM
        prompt = f"""
        As a security analyst, analyze this {language} code for the vulnerability {row['CWE id and name']}.
        Application: {row['ApplicationName']}
        File: {file_name}
        Line number with potential issue: {line_number}

        CODE:
        {row['Code']}

        Focus on line {line_number} and determine if this is a true {row['CWE id and name']} vulnerability or a false positive.
        If it's a vulnerability, provide specific mitigation steps.

        Is this a false positive? Answer with 'false positive' if it is not a real vulnerability.
        If it is a real vulnerability, provide detailed mitigation steps.
        """

        try:
            # Get LLM response
            gpt_response = llm._call(prompt)
            print(f"Raw response type: {type(gpt_response)}")
            print(f"Raw response: {gpt_response}")

            # Safely handle the response regardless of its type
            mitigation_text = ""
            if isinstance(gpt_response, dict) and 'content' in gpt_response:
                mitigation_text = gpt_response['content']
            elif isinstance(gpt_response, str):
                mitigation_text = gpt_response
            else:
                mitigation_text = str(gpt_response)

            # Process response
            if "false positive" in mitigation_text.lower():
                data.at[index, 'Prediction'] = "FP"
            else:
                data.at[index, 'Prediction'] = "Vulnerability"
                data.at[index, 'Mitigation'] = mitigation_text

            print(f"Processed row {index}: {row['ApplicationName']}")

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            data.at[index, 'Prediction'] = "Error"
            data.at[index, 'Mitigation'] = f"Error: {str(e)}"

    # Export to Excel
    data.to_excel("Code_Snippet_vulns_Analysis.xlsx", index=False)
    print("Analysis complete. Results saved to Code_Snippet_vulns_Analysis.xlsx")

if __name__ == "__main__":
    main()




import pandas as pd
import os
from langchain.llms import OpenAI
from langchain.llms.base import LlamaLLM

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

        # Construct prompt for LLM
        prompt = f"""
        As a security analyst, analyze this {language} code for the vulnerability {row['CWE id and name']}.
        Application: {row['ApplicationName']}
        File: {file_name}
        Line number with potential issue: {line_number}

        CODE:
        {row['Code']}

        Focus on line {line_number} and determine if this is a true {row['CWE id and name']} vulnerability or a false positive.
        If it's a vulnerability, provide specific mitigation steps.

        Is this a false positive? Answer with 'false positive' if it is not a real vulnerability.
        If it is a real vulnerability, provide detailed mitigation steps.
        """

        try:
            # Get LLM response as a string
            gpt_response = llm._call(prompt)

            # Process response
            if "false positive" in gpt_response.lower():
                data.at[index, 'Prediction'] = "FP"
            else:
                data.at[index, 'Prediction'] = "Vulnerability"
                data.at[index, 'Mitigation'] = gpt_response

            print(f"Processed row {index}: {row['ApplicationName']}")

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            data.at[index, 'Prediction'] = "Error"
            data.at[index, 'Mitigation'] = f"Error: {str(e)}"

    # Export to Excel
    data.to_excel("Code_Snippet_vulns_Analysis.xlsx", index=False)
    print("Analysis complete. Results saved to Code_Snippet_vulns_Analysis.xlsx")

if __name__ == "__main__":
    main()






import os

# Extract file name and line number from Source
source_parts = row['Source'].split(':')
file_name = source_parts[0]
line_number = source_parts[1] if len(source_parts) > 1 else "1"

# Set default code context
code_context = ""

# Try to read the file content
try:
    # Make sure to use the correct path to your files
    file_path = os.path.join("your_code_directory", file_name)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
        code_context = f"Full file content:\n{file_content}"
    else:
        code_context = "File not found. Analysis based only on provided code snippet."
except Exception as e:
    code_context = f"Error reading file: {str(e)}"

try:
    # Get LLM response
    gpt_response = llm._call(prompt)
    
    # The response is a string, not a JSON object with 'content' key
    # So we need to use it directly
    mitigation_text = gpt_response  # Use the string response directly
    
    # Check if "false positive" appears in the response text
    if "false positive" in mitigation_text.lower():
        data.at[index, 'Prediction'] = "FP"
    else:
        data.at[index, 'Prediction'] = "Vulnerability"
        data.at[index, 'Mitigation'] = mitigation_text
    
    print(f"Processed row {index}: {row['ApplicationName']}")
    
except Exception as e:
    print(f"Error processing row {index}: {str(e)}")
    # Still try to handle the response even if there was an error
    mitigation_text = gpt_response if 'gpt_response' in locals() else "Error processing response"
    
    if isinstance(mitigation_text, str) and "false positive" in mitigation_text.lower():
        data.at[index, 'Prediction'] = "FP"
    else:
        data.at[index, 'Prediction'] = "Vulnerability"
        data.at[index, 'Mitigation'] = mitigation_text




