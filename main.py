import os
import json
import logging
from flask import Flask, Blueprint, request, Response, jsonify
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
custom_llm = Blueprint('custom_llm', __name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PROMPT_INDEX_FILE = 'prompt_indices.json'
PATHWAYS_MESSAGES_FILE = 'pathways.json'

# Ensure the JSON file exists
if not os.path.exists(PROMPT_INDEX_FILE):
    with open(PROMPT_INDEX_FILE, 'w') as f:
        json.dump({}, f)

# Load the prompt messages
with open(PATHWAYS_MESSAGES_FILE, 'r') as f:
    prompt_messages = json.load(f)

def get_prompt_index(call_id, increment=True):
    with open(PROMPT_INDEX_FILE, 'r') as f:
        indices = json.load(f)

    index = indices.get(call_id, 0)

    if increment:
        indices[call_id] = index + 1 if index + 1 < len(prompt_messages) else 0

    with open(PROMPT_INDEX_FILE, 'w') as f:
        json.dump(indices, f)

    return index

def generate_streaming_response(data):
    """
    Generator function to simulate streaming data.
    """
    for message in data:
        json_data = message.model_dump_json()
        yield f"data: {json_data}\n\n"

@custom_llm.route('/chat/completions', methods=['POST'])
def openai_advanced_custom_llm_route():
    request_data = request.get_json()
    streaming = request_data.get('stream', False)
    next_prompt = ''

    call_id = request_data['call']['id']
    prompt_index = get_prompt_index(call_id, False)

    last_assistant_message = ''
    if 'messages' in request_data and len(request_data['messages']) >= 2:
        last_assistant_message = request_data['messages'][-2]

    last_message = request_data['messages'][-1]
    pathway_prompt = prompt_messages[prompt_index]

    # Extract customer data
    customer_data = request_data.get('customer', {})
    customer_name = customer_data.get('name', '')
    customer_number = customer_data.get('number', '')
    company_name = request_data.get('company_name', '')
    industry = request_data.get('industry', '')
    email = request_data.get('email', '')
    website = request_data.get('website', '')

    # Replace variables in the pathway prompt
    next_prompt = pathway_prompt['next'].replace('{{13.`1`}}', customer_name)\
                                        .replace('{{13.`4`}}', email)\
                                        .replace('{{13.`7`}}', company_name)\
                                        .replace('{{13.`8`}}', industry)\
                                        .replace('{{13.`9`}}', website)

    if 'check' in pathway_prompt and pathway_prompt['check']:
        prompt = f"""
        You're an AI classifier. Your goal is to classify the following condition/instructions based on the last user message. If the condition is met, you only answer with a lowercase 'yes', and if it was not met, you answer with a lowercase 'no' (No Markdown or punctuation).
        ----------
        Conditions/Instructions: {pathway_prompt['check']}"""

        if last_assistant_message:
            prompt_completion_messages = [{
                "role": "system",
                "content": prompt
            }, {
                "role":
                "assistant",
                "content":
                last_assistant_message['content']
            }, {
                "role": "user",
                "content": last_message['content']
            }]
        else:
            prompt_completion_messages = [{
                "role": "system",
                "content": prompt
            }, {
                "role": "user",
                "content": last_message['content']
            }]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_completion_messages,
            max_tokens=10,
            temperature=0.7)

        if (completion.choices[0].message.content == 'yes'):
            prompt_index = get_prompt_index(call_id)
            next_prompt = pathway_prompt['next']
        else:
            next_prompt = pathway_prompt['error']
    else:
        prompt_index = get_prompt_index(call_id)
        next_prompt = pathway_prompt['next']

    modified_messages = [{
        "role": "system",
        "content": next_prompt
    }, {
        "role": "user",
        "content": last_message['content']
    }]

    request_data['messages'] = modified_messages

    del request_data['call']
    del request_data['metadata']
    del request_data['phoneNumber']
    del request_data['customer']

    if streaming:
        chat_completion_stream = client.chat.completions.create(**request_data)

        return Response(generate_streaming_response(chat_completion_stream),
                        content_type='text/event-stream')
    else:
        chat_completion = client.chat.completions.create(**request_data)
        return Response(chat_completion.model_dump_json(),
                        content_type='application/json')

app.register_blueprint(custom_llm)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
