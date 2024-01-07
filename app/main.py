from flask import Flask, request, jsonify
from chatbot import generate, clearHistory

app = Flask(__name__)


lastUserMessage = ""
lastResponse = ""

@app.route("/generate-response", methods=["POST"])
def generateResponse():
    lastUserMessage = request.get_json()["message"]
    lastResponse = generate(lastUserMessage)
    return jsonify({"bot-response": lastResponse}), 200

@app.route("/get-last-response", methods=["GET"])
def getLastResponse():
    return jsonify({"user-message": lastUserMessage, "bot-response": lastResponse}), 200

@app.route("/delete-history", methods=["DELETE"])
def deleteHistory():
    clearHistory()
    return "", 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)