# ============================================================
# WEEK 2 - TASK 02
# RESUME PARSER USING NLP
# FLASK BACKEND API
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile

from text_extractor import extract_text
from nlp_parser import parse_resume


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "AI Resume Parser API is running"
    })


# ============================================================
# RESUME PARSING API
# ============================================================

@app.route("/parse-resume", methods=["POST"])
def parse_resume_api():

    try:

        # Check file
        if "resume" not in request.files:

            return jsonify({
                "status": "error",
                "message": "No resume file uploaded"
            }), 400

        file = request.files["resume"]

        if file.filename == "":

            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400

        # Check extension
        if not allowed_file(file.filename):

            return jsonify({
                "status": "error",
                "message": "Only PDF, DOCX and TXT files are supported"
            }), 400

        # Create temporary file
        extension = os.path.splitext(
            file.filename
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            file.save(temp_file.name)

            temp_path = temp_file.name

        try:

            # Extract text
            resume_text = extract_text(
                temp_path
            )

            if not resume_text.strip():

                return jsonify({
                    "status": "error",
                    "message": "Could not extract text from resume"
                }), 400

            # Parse resume
            result = parse_resume(
                resume_text
            )

        finally:

            # Delete temporary file
            if os.path.exists(temp_path):

                os.remove(temp_path)

        # Return structured JSON
        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME PARSER API")
    print("=" * 60)

    print("\nServer starting...")
    print("URL: http://127.0.0.1:5000")
    print("Endpoint: POST /parse-resume")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )