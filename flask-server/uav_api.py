# uav_api.py
from flask import Blueprint, request, jsonify
from adaptiveUAVCoverage import AdaptiveUAVCoverageEnv
from flask_cors import CORS
import numpy as np
import threading
import time
import math
from shapely.geometry import Polygon, Point

# Create a blueprint for UAV coverage API
uav_api = Blueprint('uav_api', __name__)
CORS(uav_api, resources={r"/drone-data": {"origins": "*"}})

@uav_api.route('/drone-data', methods=['POST'])
def simulate_coverage():
    """
    API endpoint to run the UAV coverage simulation
    Expects JSON input with:
    - payload_weight: float 
    - battery_capacity: float
    - current_battery: float
    - coverage_area: list of {lon, lat} objects
    - no_fly_zones: list of lists of {lon, lat} objects
    """
    try:
        # Get input data from request
        data = request.json
        
        # Extract parameters
        payload_weight = float(data.get('payload_weight', 2.0))
        battery_capacity = float(data.get('battery_capacity', 2000))
        current_battery = float(data.get('current_battery', battery_capacity))
        
        # Convert current_battery from percentage to absolute value if needed
        if current_battery <= 100 and battery_capacity > 100:
            current_battery = (current_battery / 100) * battery_capacity
            
        coverage_area = data.get('coverage_area', [])
        no_fly_zones = data.get('no_fly_zones', [])
        
        # Wind data with default values
        wind_data = data.get('wind_data', {"speed": 5, "direction": 90})
        
        # Initialize the environment
        env = AdaptiveUAVCoverageEnv(
            coverage_area=coverage_area,
            no_fly_zones=no_fly_zones,
            wind_data=wind_data,
            payload_weight=payload_weight,
            battery_capacity=battery_capacity, 
            current_battery=current_battery
        )
        
        # Run simulation in a limited number of steps
        max_steps = 100
        results = run_simulation(env, max_steps)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def run_simulation(env, max_steps):
    """Run the UAV coverage simulation and collect data"""
    obs = env.reset()
    done = False
    step_count = 0

    # Data collection
    trajectory = [(float(env.position[0]), float(env.position[1]))]
    energy_usage_log = []
    speed_log = []

    try:
        while not done and step_count < max_steps:
            obs, reward, done, info = env.step(None)

            # Collect data
            trajectory.append((float(env.position[0]), float(env.position[1])))

            if len(env.energy_usage_log) > len(energy_usage_log):
                energy_usage_log.append({
                    "step": int(step_count + 1),
                    "energyUsed": float(env.energy_usage_log[-1])
                })

            if len(env.speed_log) > len(speed_log):
                speed_log.append({
                    "step": int(step_count + 1),
                    "speed": float(env.speed_log[-1]),
                    "coverageRadius": float(env.coverage_radius_log[-1])
                })

            step_count += 1

        # Format coverage map
        coverage_map = []
        for point, radius in env.coverage_map.items():
            coverage_map.append({
                "point": [float(point[0]), float(point[1])],
                "radius": float(radius)
            })

        coverage_percentage = float(env._calculate_coverage_percentage() * 100)

        results = {
            "coveragePercentage": coverage_percentage,
            "remainingBattery": float(env.current_battery),
            "steps": int(step_count),
            "trajectory": trajectory,
            "energyUsageLog": energy_usage_log,
            "speedLog": speed_log,
            "coverageMap": coverage_map,
            "success": bool(coverage_percentage >= 99),
            "completed": bool(done)
        }

        return results

    except Exception as e:
        print(f"Error during simulation: {e}")
        return {"error": str(e)}


# def run_simulation(env, max_steps):
#     """Run the UAV coverage simulation and collect data"""
#     obs = env.reset()
#     done = False
#     step_count = 0
    
#     # Data collection
#     trajectory = [env.position]
#     energy_usage_log = []
#     speed_log = []
#     coverage_radius_log = []
    
#     try:
#         while not done and step_count < max_steps:
#             # Use the algorithm's built-in decision making
#             obs, reward, done, info = env.step(None)
            
#             # Collect data
#             trajectory.append(env.position)
#             if len(env.energy_usage_log) > len(energy_usage_log):
#                 energy_usage_log.append({
#                     "step": step_count + 1,
#                     "energyUsed": env.energy_usage_log[-1]
#                 })
            
#             if len(env.speed_log) > len(speed_log):
#                 speed_log.append({
#                     "step": step_count + 1,
#                     "speed": env.speed_log[-1],
#                     "coverageRadius": env.coverage_radius_log[-1]
#                 })
            
#             step_count += 1
        
#         # Format coverage map for frontend
#         coverage_map = []
#         for point, radius in env.coverage_map.items():
#             coverage_map.append({
#                 "point": point,
#                 "radius": radius
#             })
        
#         coverage_percentage = env._calculate_coverage_percentage() * 100
        
#         # Create results object
#         results = {
#             "coveragePercentage": coverage_percentage,
#             "remainingBattery": env.current_battery,
#             "steps": step_count,
#             "trajectory": trajectory,
#             "energyUsageLog": energy_usage_log,
#             "speedLog": speed_log,
#             "coverageMap": coverage_map,
#             "success": coverage_percentage >= 99,
#             "completed": done
#         }
        
#         return results
        
#     except Exception as e:
#         print(f"Error during simulation: {e}")
#         return {"error": str(e)}
        
# Add this to your main Flask app like this:
'''
from flask import Flask
from uav_api import uav_api

app = Flask(__name__)
app.register_blueprint(uav_api)

if __name__ == '__main__':
    app.run(debug=True)
'''