from flask import Flask, request, jsonify
from datetime import datetime
import socket
import argparse
import os
from gradientai import Gradient

app = Flask(__name__)
base_model = None
token = 'LXkZBp5a1nGCP16xM7QieYKNz2Ns0tOW'
workspace_id = 'b5f958d9-ffd4-41bb-a492-704114e02c8e_workspace'

os.environ['GRADIENT_ACCESS_TOKEN'] = token
os.environ['GRADIENT_WORKSPACE_ID'] = workspace_id

def prepareLlamaBot():
    global base_model
    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="llama3-8b-chat")

@app.route('/')
def index():
    return "Welcome to the Flask API!"

@app.route('/chat', methods=['POST'])
def chat():
    global base_model, user_message, chat_history
    data = request.get_json()

    if 'userMessage' not in data or not isinstance(data['userMessage'], str):
        return jsonify({'error': 'userMessage must be a string'}), 400

    if 'chatHistory' not in data or not isinstance(data['chatHistory'], list):
        return jsonify({'error': 'chatHistory must be a list'}), 400

    if not all(isinstance(item, dict) and 'User' in item and 'Llama' in item for item in data['chatHistory']):
        return jsonify({'error': 'chatHistory must be a list of dictionaries with keys User and Llama'}), 400

    user_message = data['userMessage']
    chat_history = data['chatHistory']
    chat_history_str = '\n'.join([f"{item['User']} - {item['Llama']}" for item in chat_history])

    QUERY = f"[INST]GIVEN THE CHAT HISTORY:\n{chat_history_str}\nAND THE LATEST MESSAGE FROM USER:\n{user_message}\nGIVE A RESPONSE TO THE USER\n[/INST]"

    response = base_model.complete(query=QUERY, max_generated_token_count=500).generated_output

    return jsonify({'message': response}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Specify the port number')
    args = parser.parse_args()

    port_num = args.port
    print("Starting Llama bot...\n This may take a while.")
    prepareLlamaBot()
    print(f"App running on port {port_num}")
    app.run(port=port_num)