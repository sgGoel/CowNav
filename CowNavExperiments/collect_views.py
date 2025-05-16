#!/usr/bin/env python3

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


# setup the sim
def make_sim(scene_path: str,
             sensor_resolution: Tuple[int, int] = (512, 512),
             gpu_device_id: int = 0) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu_device_id

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = sensor_resolution
    rgb_spec.position = [0.0, 1.5, 0.0]           # eye height, arbitrary but fixed
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.5
    agent_cfg.sensor_specifications = [rgb_spec]

    backend_cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(backend_cfg)


def circle_poses(center: np.ndarray, radius: float,
                 n_pts: int, look_outward: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    poses = []
    for i in range(n_pts):
        θ = 2 * math.pi * i / n_pts
        pos = center + np.array([radius * math.cos(θ), 0.0, radius * math.sin(θ)])
        heading = θ + (0.0 if look_outward else math.pi)          # face center 
        rot = quat_from_angle_axis(heading, np.array([0, 1, 0]))
        poses.append((pos, rot))
    return poses


def figure8_poses(center: np.ndarray, radius: float,
                  n_pts: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    half = n_pts // 2
    poses = circle_poses(center + np.array([radius, 0, 0]), radius, half, look_outward=False)
    poses += circle_poses(center - np.array([radius, 0, 0]), radius, half, look_outward=False)
    return poses


def straight_line_poses(start: np.ndarray, end: np.ndarray,
                        n_pts: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    poses = []
    direction = end - start
    for i in range(n_pts):
        t = i / (n_pts - 1)
        pos = start + t * direction
        heading = math.atan2(direction[0], direction[2]) + math.pi  # face forward
        rot = quat_from_angle_axis(heading, np.array([0, 1, 0]))
        poses.append((pos, rot))
    # return + reverse (for reverse sweep)
    return poses + poses[::-1]


def zig_poses(bottom_left: np.ndarray, top_right: np.ndarray,
              steps_diag: int, steps_edge: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    diag = straight_line_poses(bottom_left, top_right, steps_diag)
    edge_start = top_right
    edge_end = np.array([top_right[0], top_right[1], bottom_left[2]])
    edge = straight_line_poses(edge_start, edge_end, steps_edge)
    return diag + edge


# Auto-scaling trajectory generation based on scene size
def generate_trajectories_for_scene(bounds):
    min_bounds, max_bounds = bounds
    center = (min_bounds + max_bounds) / 2.0
    
    # Calculate scene dimensions
    dimensions = max_bounds - min_bounds
    print(f"Scene dimensions: {dimensions}")
    
    # Set fixed y height - using a reasonable value based on bounds
    y_height = min_bounds[1] + 1.5  # 1.5m off the ground
    center[1] = y_height
    
    # Scale radii based on scene size (smaller of width/depth, divided by 3)
    scene_size = min(dimensions[0], dimensions[2])
    r_small = scene_size / 5.0
    r_large = scene_size / 3.0
    
    print(f"Using center: {center}")
    print(f"Using radii: small={r_small:.2f}, large={r_large:.2f}")
    
    # Create path specifications with adjusted center and radii
    path_specs = {
        "circle_in":  circle_poses(center, r_small, n_pts=36),
        "circle_out": circle_poses(center, r_large, n_pts=36),
        "figure8":    figure8_poses(center, r_small, n_pts=72),
        "straight_fb": straight_line_poses(
            center + np.array([-r_large, 0, 0]),
            center + np.array([+r_large, 0, 0]),
            n_pts=30),
        "zig": zig_poses(
            np.array([min_bounds[0] + r_small, y_height, min_bounds[2] + r_small]),
            np.array([max_bounds[0] - r_small, y_height, max_bounds[2] - r_small]),
            steps_diag=40, steps_edge=30),
    }
    
    return path_specs, center[1]  # Return the path specs and the y-height


# to get view to match to existing information
# in future work, we'll provide even less information (just a single image)
def get_random_view(sim: habitat_sim.Simulator,
                    attempts: int = 50) -> Dict[str, np.ndarray]:
    nav    = sim.pathfinder
    agent  = sim.get_agent(0)

    def ensure_rgb(img: np.ndarray) -> np.ndarray:
        # rgba to rgb as needed
        if img.ndim == 3 and img.shape[2] == 4:
            return img[:, :, :3]
        return img

    # try up to attempts times to get an RGB frame with features in it
    for _ in range(attempts):
        # pick a navigable point (higher likelihood of having nice features)
        pos = nav.get_random_navigable_point()
        heading = np.random.uniform(0.0, 2 * np.pi)
        rot = quat_from_angle_axis(heading, np.array([0, 1, 0]))

        state = habitat_sim.AgentState()
        state.position = pos
        state.rotation = rot
        agent.set_state(state, True)

        obs = sim.get_sensor_observations()
        rgb = ensure_rgb(obs["rgb"])     

        # quick check: is the image essentially empty?
        # we're only generating one image, so it really needs to have features
        if rgb.std() < 5:        # very low contrast => probably useless
            continue

        return obs               # lgtm!

    raise RuntimeError("Couldn’t sample a valid random view in "
                       f"{attempts} attempts.")

# trajectory collection functionality
def main():
    # Set the scene path - update this to your scene
    scene = "/mnt/sdc/sgo/cownav/GaussianNavigation/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    sim = make_sim(scene_path=scene)
    
    # Get pathfinder for navigation - we'll just use it to get bounds
    nav = sim.pathfinder
    
    # Get scene bounds - either from navmesh or from scene bounds
    if nav.is_loaded:
        print(f"Navmesh loaded, using navmesh bounds")
        bounds = nav.get_bounds()
        min_bounds, max_bounds = np.array(bounds[0]), np.array(bounds[1])
    else:
        # If no navmesh, try to get scene bounds from simulator
        print("No navmesh found, trying to estimate scene bounds")
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        min_bounds = scene_bb.min
        max_bounds = scene_bb.max
        
    print(f"Scene bounds: min={min_bounds}, max={max_bounds}")
    
    # Generate trajectories based on scene size
    path_specs, y_height = generate_trajectories_for_scene((min_bounds, max_bounds))
    
    # build the trajectories - now ignoring navigability
    trajectories: Dict[str, List[habitat_sim.agent.Observations]] = {}

    agent = sim.get_agent(0)
    for name, poses in path_specs.items():
        observations: List[Dict] = []
        print(f"Processing trajectory: {name}")
        
        for i, (pos, rot) in enumerate(poses):
            # Set position with our y-height
            pos_with_y = np.copy(pos)
            pos_with_y[1] = y_height
            
            # Set agent state - ignore navigability
            agent_state = habitat_sim.AgentState()
            agent_state.position = pos_with_y
            agent_state.rotation = rot
            
            # This is where the magic happens - force setting of position even if not navigable
            agent.set_state(agent_state, True)  # Force position setting
            
            # Get observations
            obs = sim.get_sensor_observations()
            observations.append(obs)
            
            # Progress reporting
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Collected {i+1}/{len(poses)} positions")
                
        trajectories[name] = observations
        print(f"Collected {len(observations):>3} frames for {name}")

    sim.close()

    # Save the trajectories
    os.makedirs("trajectories", exist_ok=True)
    np.savez_compressed("trajectories/skokloster-castle.npz", **trajectories)
    print(f"Saved trajectories!")


if __name__ == "__main__":
    main()

