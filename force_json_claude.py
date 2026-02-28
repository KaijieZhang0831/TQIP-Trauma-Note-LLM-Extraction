import boto3
import json

client = boto3.client('bedrock-runtime', region_name='us-west-2')

LLM_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0" 

my_schema = {
    "type": "object",
    "properties": {
        "user_intent": {
            "type": "string", 
            "enum": ["purchase", "support", "inquiry"]
        },
        "confidence_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        }
    },
    "required": ["user_intent", "confidence_score"]
}

response = client.converse(
    modelId=LLM_MODEL_ID,
    messages=[{
        "role": "user",
        "content": [{"text": "Analyze this user query: 'Give me a username -> real name pair'"}]
    }],
    toolConfig={
        "tools": [
            {
                "toolSpec": {
                    "name": "extract_query_info",
                    "description": "Extract the required JSON fields from the user query.",
                    "inputSchema": {
                        "json": my_schema
                    }
                }
            }
        ],
        "toolChoice": {
            "tool": {
                "name": "extract_query_info"
            }
        }
    }
)

# Extract the guaranteed JSON from the tool's input parameters
for content in response['output']['message']['content']:
    if 'toolUse' in content:
        generated_json = content['toolUse']['input']
        print(json.dumps(generated_json, indent=2))