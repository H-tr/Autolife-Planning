import socket
from flask import Flask, request, jsonify
from autolife_planning.agents import chat_agent, point_agent

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint called by tomwebrtc when user types text"""
    try:
        data = request.json
        text = data.get("text", "")
        if text:
            # Trigger your robot logic
            chat_agent.process_chat_command(text)
            return jsonify({"status": "received"}), 200
        return jsonify({"error": "No text provided"}), 400
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"error": "Internal Error"}), 500

@app.route('/point', methods=['POST'])
def point():
    """Endpoint called by tomwebrtc when user clicks video"""
    try:
        data = request.json
        x, y = None, None

        # Parse standard JSON formats
        if "point" in data:
            # Format: {"point": "0.5,0.5"}
            parts = data["point"].split(',')
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
        elif "x" in data and "y" in data:
            # Format: {"x": 0.5, "y": 0.5}
            x, y = float(data["x"]), float(data["y"])

        if x is not None and y is not None:
            # Trigger your robot logic
            point_agent.process_point_click(x, y)
            return jsonify({"status": "received", "point": [x, y]}), 200

        return jsonify({"error": "Invalid point data"}), 400
    except Exception as e:
        print(f"Error processing point: {e}")
        return jsonify({"error": "Internal Error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify("OK"), 200
