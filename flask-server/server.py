from flask import Flask, request, jsonify
from flask_cors import CORS
from uav_api import uav_api

app = Flask(__name__)
CORS(app, resources={r"/drone-data": {"origins": "http://localhost:3000"}})  # Allow requests only from your React app
app.register_blueprint(uav_api)

# @app.route('/drone-data', methods=['POST', 'OPTIONS'])
# def receive_drone_data():
#     if request.method == 'OPTIONS':  # Handle preflight request
#         return _build_cors_preflight_response()
    
#     try:
#         data = request.json
#         print("Received Drone Data:", data)
#         return jsonify({"message": "Data received successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# def _build_cors_preflight_response():
#     response = jsonify({"message": "CORS preflight request success"})
#     response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
#     response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
#     response.headers.add("Access-Control-Allow-Headers", "Content-Type")
#     return response

if __name__ == '__main__':
    app.run(debug=True)
