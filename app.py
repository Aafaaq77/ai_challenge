"""
This script creates a Flask API that can be used to predict the topic of a text using a BERTopic
The model is loaded from the Azure ML workspace and the prediction is returned as a JSON object.
The API expects a JSON object with the key 'text' and the text to predict as value.

Example:
{
    "text": "This is a test text."
}

The API returns a JSON object with the
- original text
- the most likely topic as an integer
- the probabilities for all topics as a list

Example:
{
    "text": "This is a test text.",
    "most_likely_topic": 0,
    "topic_probabilities": [0.1, 0.2, 0.3, 0.4, ...]
}

"""

from flask import Flask, request, jsonify
from azureml.core import Workspace, Model
import joblib
import os
from bertopic import BERTopic
import re

# Flask-App
app = Flask(__name__)

# name for azure workspace and model
workspace_name = 'your_workspace_name'
subscription_id = 'your_subscription_id'
resource_group = 'your_resource_group'

# loading the workspace
ws = Workspace(
    subscription_id,
    resource_group,
    workspace_name
    )

# laoding the model registered in the workspace
model_path = Model.get_model_path('bertopic_model', _workspace=ws)
model = joblib.load(model_path)

# if preprocessing is needed
def preprocess_text(text):
    # simple preprocessing examples for text data
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    return text

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']

        processed_text = preprocess_text(text)

        # predict topic for the text and get probabilities for all topics
        topics, probabilities = model.transform([processed_text])

        # prepare the result as a JSON object to return
        result = {
            "text": text,
            "most_likely_topic": int(topics[0]),
            "topic_probabilities": probabilities[0].tolist()
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
