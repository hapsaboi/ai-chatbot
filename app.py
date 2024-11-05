from flask import Flask, request, jsonify
import logging
from datetime import datetime
import time
from flask_cors import CORS

# Import chatbot class or functions from the uploaded file
from university_chatbot import UniversityChatbot  # Adjust this to match the class or main function name

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

# Initialize chatbot instance
chatbot = UniversityChatbot()  # Assuming the class is named UniversityChatbot

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get("query")

    # Validate input
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Generate response from chatbot
    response_data = chatbot.get_response(query)
    
    return jsonify(response_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)