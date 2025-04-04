import gym
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
import random
import time
import math
from collections import deque
import heapq

class AdaptiveUAVCoverageEnv(gym.Env):
    def __init__(self, coverage_area, no_fly_zones, wind_data, payload_weight, battery_capacity, current_battery):
        super(AdaptiveUAVCoverageEnv, self).__init__()
        print("Initializing Adaptive UAV Environment...")

        # Convert coverage area and no-fly zones into polygons
        self.coverage_area = Polygon([(p["lon"], p["lat"]) for p in coverage_area])
        self.no_fly_zones = [Polygon([(p["lon"], p["lat"]) for p in zone]) for zone in no_fly_zones]

        # Wind data
        self.wind_data = wind_data

        self.energy_estimates = []

        # Calculate area dimensions and area
        minx, miny, maxx, maxy = self.coverage_area.bounds
        self.width = maxx - minx
        self.height = maxy - miny
        self.area = self.coverage_area.area

        # UAV Parameters
        self.payload_weight = payload_weight
        self.battery_capacity = battery_capacity
        self.current_battery = current_battery
        self.initial_battery = current_battery

        # Dynamic parameters
        self.energy_per_area_unit = None  # Will calculate after first movements
        self.coverage_radius = None       # Dynamic coverage radius

        # Initialize base grid size
        self.base_grid_size = min(self.width, self.height) / 10
        self.min_grid_size = self.base_grid_size / 5
        self.max_grid_size = self.base_grid_size * 2

        # Initialize UAV speed
        self.uav_speed = self._compute_dynamic_speed()

        # Generate initial grid with base size
        self.grid_size = self.base_grid_size
        self.grid = self._generate_grid(self.grid_size)

        # Make sure we have at least one point
        if len(self.grid) == 0:
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            print(f"WARNING: No valid grid points generated. Adding center point at ({center_x}, {center_y}).")
            self.grid.append((center_x, center_y))

        # Initialize coverage tracking
        self.position = self.grid[0]
        self.visited = set([self.position])
        self.coverage_map = {}  # Maps points to their coverage radius
        self.coverage_map[self.position] = self._get_current_coverage_radius()

        # Visualization data
        self.trajectory = [self.position]
        self.energy_usage_log = []
        self.speed_log = []
        self.coverage_radius_log = []

        # Debug counters
        self.stuck_count = 0
        self.last_position = None
        self.last_coverage_percentage = 0
        self.energy_estimates = []

        # Set up observation and action spaces
        self.action_space = gym.spaces.Discrete(8)  # 8 directions
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Algorithm settings
        self.adaptive_step_size = True
        self.dynamic_speed = True
        self.energy_reserve_percentage = 0.1  # Keep 10% battery reserve for emergencies

        print(f"Environment initialized with {len(self.grid)} grid points.")
        print(f"Coverage area: {self.area:.6f} square units")
        print(f"Initial grid size: {self.grid_size:.6f}")

    def _compute_dynamic_speed(self):
        """Adjust speed based on battery level, payload, and wind resistance."""
        base_speed = 10

        # Battery factor (higher speed when more battery available)
        battery_factor = 0.7 + (0.3 * self.current_battery / self.battery_capacity)

        # Payload factor (lower speed with heavier payload)
        payload_factor = max(0.5, 1 - (self.payload_weight / 20))

        # Wind effect
        wind_effect = self._compute_wind_effect()
        wind_factor = max(0.5, 1 - wind_effect)

        # Calculate speed
        speed = base_speed * battery_factor * payload_factor * wind_factor

        return max(2, min(15, speed))

    def _compute_wind_effect(self):
        """Calculate wind effect based on current conditions."""
        wind_direction = np.deg2rad(self.wind_data["direction"])
        move_angle = np.arctan2(1, 1)  # Default angle
        wind_effect = np.cos(wind_direction - move_angle) * self.wind_data["speed"] / 10
        return abs(wind_effect)

    def _generate_grid(self, grid_size):
        """Generate a grid within the coverage area, avoiding no-fly zones."""
        minx, miny, maxx, maxy = self.coverage_area.bounds
        grid = []
        x, y = minx, miny

        while y <= maxy:
            while x <= maxx:
                point = Point(x, y)
                if self.coverage_area.contains(point) and not any(zone.contains(point) for zone in self.no_fly_zones):
                    grid.append((x, y))
                x += grid_size
            x = minx
            y += grid_size

        print(f"Generated {len(grid)} grid points with grid size {grid_size}")
        return grid

    def _get_current_coverage_radius(self):
        """Calculate the coverage radius based on current energy and remaining area."""
        # Base coverage radius
        base_radius = self.grid_size / 2

        # Adjust radius based on battery percentage
        battery_percentage = self.current_battery / self.battery_capacity

        # If we have energy estimates, use them to predict needed radius
        if self.energy_estimates:
            avg_energy_per_unit = sum(self.energy_estimates) / len(self.energy_estimates)
            remaining_area = self.area * (1 - self._calculate_coverage_percentage())
            remaining_energy = self.current_battery - (self.battery_capacity * self.energy_reserve_percentage)

            # If we have plenty of energy, use larger coverage radius
            if remaining_energy > avg_energy_per_unit * remaining_area:
                adaptive_radius = base_radius * (1 + battery_percentage)
            else:
                # Calculate minimum radius needed to cover remaining area with available energy
                adaptive_radius = max(
                    self.min_grid_size / 2,
                    base_radius * min(1.0, remaining_energy / (avg_energy_per_unit * remaining_area))
                )
        else:
            # No energy estimates yet, use battery percentage
            adaptive_radius = base_radius * max(0.5, min(2.0, battery_percentage * 2))

        return max(self.min_grid_size / 2, min(adaptive_radius, self.max_grid_size))

    def _calculate_coverage_percentage(self):
        """Calculate the current coverage percentage."""
        if not self.coverage_map:
            return 0

        # Create a buffer around each visited point based on its coverage radius
        covered_area = None
        for point, radius in self.coverage_map.items():
            point_obj = Point(point)
            buffer = point_obj.buffer(radius)

            if covered_area is None:
                covered_area = buffer
            else:
                covered_area = covered_area.union(buffer)

        # Calculate intersection with coverage area to get actual coverage
        if covered_area:
            coverage_intersection = self.coverage_area.intersection(covered_area)
            return min(1.0, coverage_intersection.area / self.area)
        return 0

    def _is_point_covered(self, point):
        """Check if a point is already covered by existing coverage."""
        point_obj = Point(point)

        for visited_point, radius in self.coverage_map.items():
            visited_point_obj = Point(visited_point)
            if point_obj.distance(visited_point_obj) <= radius:
                return True

        return False

    def _find_best_next_point(self):
        """Find the best next point to visit for optimal coverage."""
        coverage_percentage = self._calculate_coverage_percentage()

        # If we're already at 100% coverage, return None
        if coverage_percentage >= 0.99:
            return None

        # Get current position
        current_x, current_y = self.position

        # Calculate current coverage radius
        current_radius = self._get_current_coverage_radius()

        # Generate potential next positions based on current grid size
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]

        # Adaptive step size based on current energy and coverage
        if self.adaptive_step_size:
            # Smaller steps when battery is low or coverage is high
            step_multiplier = max(0.5, min(2.0, self.current_battery / self.battery_capacity))

            # Adjust step size based on coverage percentage
            if coverage_percentage > 0.8:
                step_multiplier *= 0.5  # Use smaller steps for fine-tuning at high coverage
        else:
            step_multiplier = 1.0

        # Step size is determined by coverage radius and step multiplier
        step_size = current_radius * 1.5 * step_multiplier

        # Calculate potential next positions
        candidates = []
        for dx, dy in directions:
            next_x = current_x + dx * step_size
            next_y = current_y + dy * step_size
            next_point = (next_x, next_y)

            # Check if the point is valid (inside coverage area, outside no-fly zones)
            point_obj = Point(next_point)
            if not self.coverage_area.contains(point_obj) or any(zone.contains(point_obj) for zone in self.no_fly_zones):
                continue

            # Calculate energy required to move to this point
            energy_required = self._compute_energy_usage(self.position, next_point)

            # Skip if we don't have enough energy
            if energy_required >= self.current_battery:
                continue

            # Calculate how much new area this point would cover
            coverage_gain = self._estimate_coverage_gain(next_point, current_radius)

            # Calculate a score based on energy efficiency and coverage gain
            if coverage_gain > 0:
                score = coverage_gain / energy_required
            else:
                score = 0

            candidates.append((next_point, score, energy_required, coverage_gain))

        # If we couldn't find any valid next points with the current step size,
        # try with a smaller step size (half the current one)
        if not candidates and step_size > self.min_grid_size:
            reduced_step_size = max(self.min_grid_size, step_size / 2)
            for dx, dy in directions:
                next_x = current_x + dx * reduced_step_size
                next_y = current_y + dy * reduced_step_size
                next_point = (next_x, next_y)

                # Check if the point is valid
                point_obj = Point(next_point)
                if not self.coverage_area.contains(point_obj) or any(zone.contains(point_obj) for zone in self.no_fly_zones):
                    continue

                energy_required = self._compute_energy_usage(self.position, next_point)

                if energy_required >= self.current_battery:
                    continue

                coverage_gain = self._estimate_coverage_gain(next_point, current_radius)

                if coverage_gain > 0:
                    score = coverage_gain / energy_required
                else:
                    score = 0

                candidates.append((next_point, score, energy_required, coverage_gain))

        # If we still can't find any valid next points, try desperate measures
        if not candidates:
            print("WARNING: Couldn't find any valid next points. Trying last resort measures.")

            # Find uncovered areas and try to reach them
            min_distance = float('inf')
            best_uncovered_point = None

            # Sample points throughout the coverage area
            sample_size = max(100, int(self.area * 10000))  # Adjust sampling density
            minx, miny, maxx, maxy = self.coverage_area.bounds

            for _ in range(sample_size):
                # Generate random point within bounds
                rand_x = random.uniform(minx, maxx)
                rand_y = random.uniform(miny, maxy)
                sample_point = (rand_x, rand_y)

                # Check if point is valid and not already covered
                point_obj = Point(sample_point)
                if not self.coverage_area.contains(point_obj) or any(zone.contains(point_obj) for zone in self.no_fly_zones):
                    continue

                if self._is_point_covered(sample_point):
                    continue

                # Calculate distance to this point
                distance = math.sqrt((self.position[0] - sample_point[0])**2 + (self.position[1] - sample_point[1])**2)

                # Calculate energy required
                energy_required = self._compute_energy_usage(self.position, sample_point)

                # If we can reach it with our energy, consider it
                if energy_required < self.current_battery and distance < min_distance:
                    min_distance = distance
                    best_uncovered_point = sample_point

            # If we found an uncovered point we can reach, use it
            if best_uncovered_point:
                energy_required = self._compute_energy_usage(self.position, best_uncovered_point)
                coverage_gain = self._estimate_coverage_gain(best_uncovered_point, current_radius)

                if coverage_gain > 0:
                    score = coverage_gain / energy_required
                else:
                    score = 0

                candidates.append((best_uncovered_point, score, energy_required, coverage_gain))

        # Sort candidates by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return the best candidate, or None if there are no valid candidates
        return candidates[0][0] if candidates else None

    def _estimate_coverage_gain(self, point, radius):
        """Estimate how much new area would be covered by visiting this point."""
        point_obj = Point(point)
        new_buffer = point_obj.buffer(radius)

        # Calculate current covered area
        current_covered_area = None
        for visited_point, visited_radius in self.coverage_map.items():
            visited_point_obj = Point(visited_point)
            buffer = visited_point_obj.buffer(visited_radius)

            if current_covered_area is None:
                current_covered_area = buffer
            else:
                current_covered_area = current_covered_area.union(buffer)

        if current_covered_area is None:
            # Nothing covered yet, so this is all new coverage
            intersection_with_coverage_area = self.coverage_area.intersection(new_buffer)
            return intersection_with_coverage_area.area
        else:
            # Calculate the new area this point would add
            combined_area = current_covered_area.union(new_buffer)
            new_area = combined_area.area - current_covered_area.area

            # Ensure we only count area within the coverage boundary
            intersection_with_coverage_area = self.coverage_area.intersection(combined_area)
            current_intersection = self.coverage_area.intersection(current_covered_area)

            return intersection_with_coverage_area.area - current_intersection.area

    def _compute_energy_usage(self, from_pos, to_pos):
        """Calculate energy usage for movement between two positions."""
        # Calculate distance
        distance = math.sqrt((from_pos[0] - to_pos[0])**2 + (from_pos[1] - to_pos[1])**2)
        if distance == 0:
            return 0

        # Direction of movement
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        move_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0

        # Wind effect based on movement direction
        wind_direction = np.deg2rad(self.wind_data["direction"])
        wind_effect = math.cos(wind_direction - move_angle) * self.wind_data["speed"] / 10

        # Base power usage
        power_usage = 100 + (self.payload_weight * 0.5)

        # Wind resistance factor
        wind_resistance = 1 + abs(wind_effect) * 0.3

        # Speed factor (higher speed = more energy usage)
        speed_factor = self.uav_speed / 10

        # Distance factor
        distance_factor = distance / self.grid_size

        # Calculate final energy usage
        energy_usage = power_usage * wind_resistance * speed_factor * distance_factor

        # Store energy usage per unit distance for future estimates
        if distance > 0:
            energy_per_unit = energy_usage / distance
            self.energy_estimates.append(energy_per_unit)
            # Keep only the last 20 estimates
            if len(self.energy_estimates) > 20:
                self.energy_estimates = self.energy_estimates[-20:]

        return energy_usage

    def step(self, action=None):
        """Move UAV to next position for optimal coverage."""
        # Update UAV speed if dynamic speed is enabled
        if self.dynamic_speed:
            self.uav_speed = self._compute_dynamic_speed()

        # Find the best next point to visit
        next_position = self._find_best_next_point()

        # If no valid next point found, we're either done or out of energy
        if next_position is None:
            coverage_percentage = self._calculate_coverage_percentage()
            print(f"No valid next position found. Coverage: {coverage_percentage*100:.1f}%")

            # Check if we have satisfactory coverage
            if coverage_percentage >= 0.99:
                print("Achieved full coverage! Mission complete.")
                return self._get_observation(), 10, True, {"success": True}
            else:
                # Check if we're out of energy
                if self.current_battery < self.battery_capacity * 0.1:
                    print("Low battery! Unable to complete coverage.")
                    return self._get_observation(), -5, True, {"success": False, "reason": "low_battery"}
                else:
                    print("Unable to find path to uncovered areas.")
                    return self._get_observation(), -5, True, {"success": False, "reason": "no_path"}

        # Calculate energy for this move
        energy_used = self._compute_energy_usage(self.position, next_position)

        # Check if we have enough energy
        if energy_used <= self.current_battery:
            prev_position = self.position
            self.position = next_position
            self.trajectory.append(self.position)

            # Update coverage map with this position and its coverage radius
            current_radius = self._get_current_coverage_radius()
            self.coverage_map[self.position] = current_radius

            # Update tracking data
            self.visited.add(self.position)
            self.current_battery -= energy_used

            # Log data for visualization
            self.energy_usage_log.append(energy_used)
            self.speed_log.append(self.uav_speed)
            self.coverage_radius_log.append(current_radius)

            # Calculate reward
            coverage_percentage = self._calculate_coverage_percentage()
            coverage_gain = coverage_percentage - self.last_coverage_percentage
            self.last_coverage_percentage = coverage_percentage

            reward = coverage_gain * 100  # Scale reward based on coverage gain

            print(f"Step: Moved from {prev_position} to {next_position}")
            print(f"Energy used: {energy_used:.2f}, remaining: {self.current_battery:.2f}")
            print(f"Coverage: {coverage_percentage*100:.1f}%")
            print(f"Coverage radius: {current_radius:.4f}")
        else:
            # Not enough energy
            print(f"Not enough energy to move to {next_position}.")
            print(f"Required: {energy_used:.2f}, Available: {self.current_battery:.2f}")
            reward = -1

        # Check if done (full coverage or out of energy)
        done = (self._calculate_coverage_percentage() >= 0.99) or (self.current_battery <= 0)

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Get current observation state."""
        coverage_percentage = self._calculate_coverage_percentage()

        return np.array([
            self.position[0],
            self.position[1],
            self.current_battery / self.battery_capacity,
            self.uav_speed / 15,
            coverage_percentage,
            self._get_current_coverage_radius() / self.max_grid_size
        ], dtype=np.float32)

    def reset(self):
        """Reset UAV for a new mission."""
        # Reset UAV state
        minx, miny, maxx, maxy = self.coverage_area.bounds
        self.position = self.grid[0]
        self.visited = set([self.position])
        self.trajectory = [self.position]
        self.current_battery = self.initial_battery
        self.uav_speed = self._compute_dynamic_speed()

        # Reset coverage tracking
        current_radius = self._get_current_coverage_radius()
        self.coverage_map = {self.position: current_radius}
        self.last_coverage_percentage = 0

        # Reset logs
        self.energy_usage_log = []
        self.speed_log = []
        self.coverage_radius_log = []
        self.energy_estimates = []

        return self._get_observation()

    def render(self, mode='human'):
        """Visualize UAV coverage and path."""
        plt.figure(figsize=(12, 10))

        # Create subplots: main plot and resource usage
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        # Main coverage plot
        # Plot coverage area boundary
        x, y = self.coverage_area.exterior.xy
        ax1.plot(x, y, c="green", linewidth=2, label="Coverage Boundary")

        # Plot no-fly zones
        for i, zone in enumerate(self.no_fly_zones):
            x, y = zone.exterior.xy
            label = "No-Fly Zone" if i == 0 else ""
            ax1.plot(x, y, c="red", linestyle='--', linewidth=2, label=label)

        # Plot the coverage circles for each visited point
        for point, radius in self.coverage_map.items():
            circle = plt.Circle(point, radius, color='blue', fill=True, alpha=0.1)
            ax1.add_patch(circle)

        # Plot the trajectory path
        if len(self.trajectory) > 1:
            trajectory_x, trajectory_y = zip(*self.trajectory)
            ax1.plot(trajectory_x, trajectory_y, 'b-', linewidth=1.5, alpha=0.6, label="UAV Path")

        # Plot current position
        ax1.scatter(*self.position, c="red", s=100, label="UAV Position")

        # Set title with info
        coverage_percentage = self._calculate_coverage_percentage()
        ax1.set_title(f"UAV Coverage Mission | Coverage: {coverage_percentage*100:.1f}%\n"
                     f"Battery: {self.current_battery:.1f}/{self.battery_capacity} | "
                     f"Speed: {self.uav_speed:.2f} m/s | "
                     f"Radius: {self._get_current_coverage_radius():.4f}")

        ax1.set_aspect('equal')

        # Energy usage plot
        steps = range(1, len(self.energy_usage_log) + 1)
        if self.energy_usage_log:
            ax2.plot(steps, self.energy_usage_log, 'r-', label="Energy Usage")
            ax2.set_ylabel("Energy Used")
            ax2.set_title("Energy Consumption per Step")
            ax2.grid(True)

        # Speed and coverage radius plot
        if self.speed_log and self.coverage_radius_log:
            ax3.plot(steps, self.speed_log, 'g-', label="Speed")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Speed (m/s)")
            ax3.grid(True)

            # Twin axis for coverage radius
            ax3b = ax3.twinx()
            ax3b.plot(steps, self.coverage_radius_log, 'b--', label="Coverage Radius")
            ax3b.set_ylabel("Coverage Radius")

            # Combine legends
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            ax3.set_title("Speed and Coverage Radius per Step")

        # Handle legend for main plot
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()
        plt.pause(0.2)
        plt.close()