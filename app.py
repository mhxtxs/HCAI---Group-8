import os
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder=".", static_url_path="")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are an assistant called FlowerLens. "
    "You help users identify and understand all kinds of flowers. "
    "When the user uploads a flower image, you will usually receive a message like "
    "'I just uploaded a picture of a flower' together with some KNN settings. "
    "Always pick one concrete, likely flower type (for example: rose, tulip, sunflower, daisy, orchid, iris, etc.) "
    "and give a short explanation of why you guessed this type. "
)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "No message"}), 400

    user_message = data["message"]

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=400,
        temperature=0.8,
    )

    answer = resp.choices[0].message.content
    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
