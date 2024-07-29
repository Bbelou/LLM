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

# Initialize sessions to store context
sessions = {}

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
        yield f"data: {message['choices'][0]['delta']['content']}\n\n"

def check_condition(prompt, user_response):
    if 'check' in prompt:
        condition_prompt = f"You're an AI classifier. {prompt['check']}"
        classifier_input = user_response
        
        return classify_response(condition_prompt, classifier_input)
    return True

def classify_response(condition_prompt, user_response):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": condition_prompt},
                  {"role": "user", "content": user_response}],
        max_tokens=10
    )
    return completion.choices[0].message.content == 'yes'

@custom_llm.route('/chat/completions', methods=['POST'])
def openai_advanced_custom_llm_route():
    request_data = request.get_json()
    streaming = request_data.get('stream', False)
    call_id = request_data['call']['id']

    # Initialize session if not already present
    if call_id not in sessions:
        sessions[call_id] = {
            'customer_name': '',
            'email': '',
            'company_name': ''
        }

    # Extract customer data from request
    customer_data = request_data.get('customer', {})
    sessions[call_id]['customer_name'] = customer_data.get('name', '')
    sessions[call_id]['email'] = request_data.get('email', '')
    sessions[call_id]['company_name'] = customer_data.get('company_name', '')

    last_message = request_data['messages'][-1]
    prompt_index = get_prompt_index(call_id, False)
    pathway_prompt = prompt_messages[prompt_index]

    # Replace variables in the pathway prompt
    next_prompt = pathway_prompt['next'].replace('{{13.`1`}}', sessions[call_id]['customer_name'])\
                                          .replace('{{13.`4`}}', sessions[call_id]['email'])\
                                          .replace('{{13.`7`}}', sessions[call_id]['company_name'])

    if check_condition(pathway_prompt, last_message['content']):
        # Create the modified messages for the model
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
            # Handle streaming response
            chat_completion_stream = client.chat.completions.create(**request_data)

            return Response(generate_streaming_response(chat_completion_stream), content_type='text/event-stream')
        else:
            # Handle non-streaming response
            chat_completion = client.chat.completions.create(**request_data)
            response = chat_completion.choices[0].message.content  # Adjusted to access content directly

            return jsonify({'content': response})
    else:
        next_prompt = pathway_prompt.get('error', 'Sorry, I didn’t quite catch that. Could you repeat?')

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
            # Handle error streaming response
            chat_completion_stream = client.chat.completions.create(**request_data)

            return Response(generate_streaming_response(chat_completion_stream), content_type='text/event-stream')
        else:
            chat_completion = client.chat.completions.create(**request_data)
            error_response = chat_completion.choices[0].message.content
            
            return jsonify({'error': error_response})

app.register_blueprint(custom_llm)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
