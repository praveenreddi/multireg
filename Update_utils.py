# 1. Update your system message function
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
