
# experiment 1:

# collect images from path (store as habitat_dict views)

# gaussian splatting --> construct additional views (this is the matching aware step)
# (skipping for now, it's a future extension of the project)

# compute "information density" of that path

# experiment 2:

compute_image_pair_similarity(self, obs1, obs2, resize_factors=4) # for goal object and a bunch of candidate views
# NOTE: goal object should also be a habitat_dict view, and similarity score should be 0
# show relationship between information density and best image pair similarity (ie, high information density yields high image similarity scores)


# TODO: visually verify that the trajectory views are good (and do not include the inside of a chair, or something)

# TODO: make sure all our experiment code only samples 30 frames from each trajectory



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


# was running into an issue with 0 frames being collected, solution is to be less strict with teh trajectory
def find_navigable_point_near(nav, point, max_search_radius=5.0, step=0.5):
    """Find a navigable point near the requested point."""
    original_y = point[1]  
    
    # can we get here?
    if nav.is_navigable(point):
        print(f"debug: could navigate to {point} right off the bat.")
        return point
    
    # search in expanding circles
    for radius in np.arange(step, max_search_radius + step, step):
        for angle in np.linspace(0, 2*math.pi, num=16, endpoint=False):
            test_point = np.array([
                point[0] + radius * math.cos(angle),
                original_y,  
                point[2] + radius * math.sin(angle)
            ])
            if nav.is_navigable(test_point):
                print(f"Found navigable point at radius {radius:.2f}")
                return test_point
    
    # give up
    return None


# trajectory collection functionality
def main():
    scene = "/mnt/sdc/sgo/cownav/GaussianNavigation/data/scene_datasets/habitat-test-scenes/van-gogh-room.glb" #skokloster-castle.glb
    sim = make_sim(scene_path=scene)
    
    nav = sim.pathfinder
    
    # make sure the mesh loaded
    if not nav.is_loaded:
        print("Does the scene have a navmesh?")
        sim.close()
        return
    
    # print mesh ingo
    print(f"navmesh bounds: {nav.get_bounds()}")
    bounds = nav.get_bounds()
    min_bounds, max_bounds = np.array(bounds[0]), np.array(bounds[1])
    
    nav_center_raw = (min_bounds + max_bounds) / 2.0
    print(f"raw center point: {nav_center_raw}")
    
    # get a close by navigable point
    nav_center = find_navigable_point_near(nav, nav_center_raw)
    if nav_center is None:
        print("Didn't work, try manual exploration?")
        found_navigable = False
        for _ in range(100):
            x = np.random.uniform(min_bounds[0], max_bounds[0])
            z = np.random.uniform(min_bounds[2], max_bounds[2])
            y = nav_center_raw[1]  #
            point = np.array([x, y, z])
            if nav.is_navigable(point):
                nav_center = point
                found_navigable = True
                print(f"will try to go to: {nav_center}")
                break
        
        if not found_navigable:
            print("Failed to nav, exiting.")
            sim.close()
            return
    
    y_nav = nav_center[1]
    print(f"Center: {nav_center}")
    
    r_small, r_large = 1.0, 2.0 #reduced circle size, and also the range of sweeps, in hopes this would help with obstacle avoidance.
    
    # build the trajectories
    trajectories: Dict[str, List[habitat_sim.agent.Observations]] = {}

    path_specs = {
        "circle_in":  circle_poses(nav_center, r_small, n_pts=36),
        "circle_out": circle_poses(nav_center, r_large, n_pts=36),
        "figure8":    figure8_poses(nav_center, r_small, n_pts=72),
        "straight_fb": straight_line_poses(
            nav_center + np.array([-r_large/2, 0, 0]),  
            nav_center + np.array([+r_large/2, 0, 0]),
            n_pts=30),
        "zig": zig_poses(
            nav_center - np.array([r_large, 0, r_large]),  
            nav_center + np.array([r_large, 0, r_large]),
            steps_diag=40, steps_edge=30),
    }

    agent = sim.get_agent(0)
    for name, poses in path_specs.items():
        observations: List[Dict] = []
        print(f"Processing trajectory: {name}")
        valid_positions = 0
        for i, (pos, rot) in enumerate(poses):
            # snap to closest navigable point on mesh
            original_pos = np.copy(pos)
            nav_pos = original_pos.copy()
            nav_pos[1] = y_nav
            
            if not nav.is_navigable(nav_pos):
                # Try to find nearby navigable point
                nearby_nav_pos = find_navigable_point_near(nav, nav_pos, max_search_radius=1.0)
                if nearby_nav_pos is not None:
                    nav_pos = nearby_nav_pos
                else:
                    # Skip this position if no navigable point found nearby
                    continue
            
            valid_positions += 1
            agent_state = habitat_sim.AgentState()
            agent_state.position = nav_pos
            agent_state.rotation = rot
            agent.set_state(agent_state, False)
            obs = sim.get_sensor_observations()
            observations.append(obs)
            
            if valid_positions % 10 == 0:
                print(f"  Collected {valid_positions} valid positions so far...")
                
        trajectories[name] = observations
        print(f"Collected {len(observations):>3} frames for {name}")

    sim.close()

    # Only save if we have data
    total_frames = sum(len(obs) for obs in trajectories.values())
    if total_frames > 0:
        # Create directory if it doesn't exist
        os.makedirs("trajectories", exist_ok=True)
        
        # save locally!
        np.savez_compressed("trajectories/habitat_views.npz", **trajectories)
        print(f"Saved {total_frames} total frames to trajectories/habitat_views.npz")
    else:
        print("No frames were collected! Check the navmesh and scene setup.")


if __name__ == "__main__":
    main()








def build_clip_extractor(device: str = "cuda:0"):
    """
    Returns a function   f(rgb_np_uint8) -> List[int]   where each int is a
    64‑bit hash of a CLIP patch embedding.  The hash becomes the CMS token.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model.eval().requires_grad_(False).to(device)

    # We want *patch embeddings*, not just the CLS token.
    def _forward_patch(x: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of shape (num_patches, dim).  The ViT patch grid for
        224×224 with 32‑pixel patches is 7×7=49 patches.
        """
        # open_clip’s visual model gives patches if we ask for them:
        return model.visual(x, return_patch=True)[1].squeeze(0)  # (49, 768)

    def extract(rgb: np.ndarray) -> List[int]:
        im = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)  # (1,3,224,224)
        with torch.no_grad():
            patches = _forward_patch(im)          # (N_patches, 768)
            # Normalize then convert to 64‑bit hashes
            patches = F.normalize(patches, dim=-1)  # cosine / unit‑norm
            # Simple locality‑sensitive hash: sign bits → 64‑bit integer
            # We down‑project 768‑D to 64 bits with a random ±1 matrix.
            sign_bits = (patches @ extract.proj > 0).to(torch.bool)
        return _bits_to_int(sign_bits)

    # store a random ±1 projection on the function object (constant across calls)
    rng = torch.Generator(device).manual_seed(42)
    extract.proj = torch.empty((768, 64), device=device).uniform_(-1.0, 1.0).sign_()

    # helper: turn 64 boolean bits into integer tokens
    def _bits_to_int(bitmat: torch.BoolTensor) -> List[int]:
        # bitmat: (N, 64) bool
        words = bitmat.view(-1, 8, 8)             # -> (N, 8 bytes, 8 bits)
        bytes_ = words.bitwise_left_shift(torch.arange(8, device=bitmat.device))
        ints = bytes_.sum(-1).cpu().numpy().view('<u8')  # little‑endian 64‑bit
        return ints.tolist()

    return extract

