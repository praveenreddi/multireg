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




