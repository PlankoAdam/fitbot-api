from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from chatbot import processTXT,processPDF,answer,getContext
from waitress import serve
import logging

logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

app = Flask(__name__)


lastUserMessage = ""
lastResponse = ""

cors = CORS()
cors.init_app(app, resource={r"/api/*": {"origins": "*"}})

@app.route('/process', methods=['POST'])
def process():
    for fname in request.files:
        f = request.files.get(fname)
        secfname = secure_filename(fname)
        f.save('./uploads/%s' % secfname)

        if (f.content_type == 'application/pdf') | (f.content_type == 'text/plain'):
            chunks = []

            if f.content_type == 'application/pdf':
                chunks = processPDF(secfname)
            elif f.content_type == 'text/plain':
                chunks = processTXT(secfname)

            if os.path.exists('./uploads/%s' % secfname):
                os.remove('./uploads/%s' % secfname)

            return jsonify({"chunks": chunks}), 200

        return 'wrong file type', 400

@app.route('/answer', methods=['POST'])
def askQuery():
    query = request.get_json()["query"]
    chunks = request.get_json()["chunks"]
    ctx = getContext(chunks, query)
    result = answer(ctx, query)
    return jsonify({"answer": result, "context": ctx}), 200

if __name__ == "__main__":
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    serve(app, host="0.0.0.0", port=5000, threads=2)