import os
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from statistics import mean, stdev

from information_computation import *
from collect_views import *

#run export PYTHONPATH=/mnt/sdc/sgo/cownav/:/mnt/sdc/sgo/cownav/GaussianNavigation/
from GaussianNavigation.vector_env.envs.instance_imagenav_env import *

##### data & utilities #####

SCENES: List[str] = ["apartment_1", "skokloster-castle", "van-gogh-room"]
TRAJECTORIES: List[str] = [
    "circle_in",
    "circle_out",
    "figure8",
    "straight_fb",
    "zig",
]

# wherever the .glb scenes live
SCENE_ROOT = "/mnt/sdc/sgo/cownav/GaussianNavigation/data/scene_datasets/habitat-test-scenes"
TRAJ_ROOT = "trajectories"  # wherever output of collect_views lives

# setup dummy NiceEnv (so we can use compute_image_pair_similarity)
helper = SimpleNamespace()
helper.similarity_method = 1 #plain lightglue                                   
helper.device           = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
helper.extractor        = DISK(max_num_keypoints=2048).eval().to(helper.device)
helper.matcher          = LightGlue(features='disk').eval().to(helper.device)
helper.matcher.compile(mode='reduce-overhead')
helper.max_depth        = 10.0                                   # meters

compute_image_pair_similarity = NiceEnv.compute_image_pair_similarity.__get__(helper, NiceEnv)

def glb_path(scene_name):
    return f"{SCENE_ROOT}/{scene_name}.glb"



##### plotting #####

def plot_trajectories() -> None:
    # get path specs for scene (in this case skokloster-castle, doesn't matter though)
    scene = "/mnt/sdc/sgo/cownav/GaussianNavigation/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    sim = make_sim(scene_path=scene)

    nav = sim.pathfinder
    if nav.is_loaded:
        print("loaded navmesh")
        min_bounds, max_bounds = [np.array(b) for b in nav.get_bounds()]
    else:
        print("no navmesh found")
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        min_bounds, max_bounds = scene_bb.min, scene_bb.max

    print(f"bounds: min={min_bounds}, max={max_bounds}")

    path_specs, y_height = generate_trajectories_for_scene((min_bounds, max_bounds))
    print("path_specs", path_specs)

    # plot the trajectories
    out_dir = "path_plots"
    os.makedirs(out_dir, exist_ok=True)

    processed = set()
    for traj_name, poses in path_specs.items():
        if traj_name in processed:
            continue

        if len(poses) < 30:
            raise ValueError(f"Trajectory '{traj_name}' only has {len(poses)} poses – need ≥ 30.")

        idx = np.random.choice(len(poses), size=30, replace=False)
        idx.sort()

        xs, zs = [], []
        for i in idx:
            pos = poses[i][0]          # (pos, quaternion)
            xs.append(pos[0])
            zs.append(pos[2])

        plt.figure(figsize=(6, 6))
        plt.scatter(xs, zs, c="red", s=24)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()

        outfile = os.path.join(out_dir, f"{traj_name}.png")
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"saved {outfile}")

        processed.add(traj_name)
        if len(processed) == 5:
            break

    print(f"\nplots at '{out_dir}'")


def plot_entropy_bars(traces_f):
    scene = "apartment_1"
    trajectory_list = ["circle_in", "circle_out", "figure8", "straight_fb", "zig"]
    out_file=f"results/{scene}_entropy_hm.png"
    
    bar_gap=2
    pad_rows=2
    row_height=0.22
    bar_width=1.8

    traces = json.load(open(traces_f))
    cols = []
    for i, traj in enumerate(trajectory_list):
        series = traces[scene][traj]
        if len(series) != 30:
            raise ValueError(f"{scene}|{traj}: need 30 values")
        cols.append(np.array(series).reshape(-1, 1))
        if i < len(trajectory_list) - 1:
            cols.append(np.full((30, bar_gap), np.nan))

    body = np.hstack(cols)
    if pad_rows:
        pad = np.full((pad_rows, body.shape[1]), np.nan)
        data = np.vstack((pad, body, pad))
    else:
        data = body

    #cmap = LinearSegmentedColormap.from_list("reddish", ["#ffcccc", "#8b0000"])
    cmap = cm.summer.reversed()
    cmap.set_bad("white")
    norm = Normalize(vmin=np.nanmin(body), vmax=np.nanmax(body))
    tot_cols = data.shape[1]

    fig_w = bar_width * (tot_cols / (1 + bar_gap * (len(trajectory_list) - 1) / len(trajectory_list)))
    fig_h = row_height * data.shape[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_frame_on(False)
    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    total_cols_nominal = len(trajectory_list) + bar_gap * (len(trajectory_list) - 1)
    for i, traj in enumerate(trajectory_list):
        x_ax = (i * (1 + bar_gap) + 0.5) / total_cols_nominal
        #ax.text(x_ax, 0.05, f"{scene} | {traj}", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        #ax.text(x_ax, 0.95, f"{traces[scene][traj][-1]:.3f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8)

    #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, aspect=20).set_label("Entropy")
    plt.tight_layout(pad=0.4)

    plt.savefig(out_file, dpi=300)
    plt.close()


# show relationship between information density and best image pair similarity (ie, high information density yields high image similarity scores)
def plot_match_vs_entropy(
    match_scores_path,
    sketch_entropies_path,
    save_path,
):
    # load data
    with open(match_scores_path) as f:
        match_scores = json.load(f)
    with open(sketch_entropies_path) as f:
        entropies = json.load(f)

    # prep plot
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.get_cmap("tab10")           # diff col per scene
    scene_colors = {s: cmap(i) for i, s in enumerate(match_scores)}

    for scene, color in scene_colors.items():
        for traj, stats in match_scores[scene].items():
            x = entropies[scene][traj]        # entropy on x
            y = stats["mean"]                 # match on y
            ax.scatter(x, y, color=color, edgecolor="black", s=60,
                       label=scene if traj == next(iter(match_scores[scene])) else "")
            ax.annotate(traj, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)

    all_xs, all_ys = [], []
    for scene in match_scores:
        for traj in match_scores[scene]:
            all_xs.append(entropies[scene][traj])
            all_ys.append(match_scores[scene][traj]["mean"]) 

    slope, intercept = np.polyfit(all_xs, all_ys, 1)
    x_fit = np.linspace(min(all_xs), max(all_xs), 100)
    y_fit = slope * x_fit + intercept

    ax.plot(x_fit, y_fit, lw=2, color="black", ls="--",
            label=f"best-fit: y = {slope:.2f}x + {intercept:.2f}")

    # make the graph prettier
    ax.set_xlabel("Sketch entropy")
    ax.set_ylabel("Mean LightGlue match score")
    ax.set_title("Sketch entropy vs. match score per trajectory")
    ax.grid(ls="--", alpha=0.3)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = {lbl: h for lbl, h in zip(labels, handles)}
    ax.legend(by_label.values(), by_label.keys(), title="Scene", framealpha=0.9)

    fig.tight_layout()

    # save
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    else:
        plt.show()

    return fig



##### experiments #####

def build_sketches():
    scenes = ["apartment_1", "skokloster-castle", "van-gogh-room"]
    entropies = {}
    traces={}

    for sc in scenes:
        archive = np.load(f"trajectories/{sc}.npz", allow_pickle=True)
        entropies[sc] = {}
        traces[sc] = {}

        for traj_name in archive.keys():
            traj = archive[traj_name]    

            print(f"\n\nloading {sc} {traj_name}")      

            # sample just 30 frames (without replacement)
            if len(traj) < 30:
                raise ValueError(f"trajectory only has {len(traj)} frames, need at least 30 to sample.")
            idx = np.random.choice(len(traj), size=30, replace=False)
            idx.sort()                      
            views = [traj[i]['rgb'] for i in idx]

            extractor = build_lightglue_extractor()
            sketch = InformationSketch()
            path_ent  = []

            for i, rgb in enumerate(views, 1):
                sketch.update(extractor(rgb))
                path_ent.append(sketch.entropy())
                print(f"frame {i:03d}, total entropy of path = {sketch.entropy():.3f}")

            entropies[sc][traj_name] = sketch.entropy()
            traces[sc][traj_name] = path_ent

    return entropies, traces

def compute_match_scores() -> Dict[str, Dict[str, Dict[str, float]]]:

    results = {}
    sims = {sc: make_sim(str(glb_path(sc))) for sc in SCENES}

    for sc in SCENES:
        archive = np.load(f"{TRAJ_ROOT}/{sc}.npz", allow_pickle=True)
        sim = sims[sc]
        results[sc] = {}

        for traj_name in TRAJECTORIES:
            traj_views = archive[traj_name]
            best_scores= []

            print(f"at {sc}, {traj_name}")
            # get 5 random views
            for _ in range(5):
                random_obs = get_random_view(sim)  

                best = -100
                for obs in traj_views:
                    score = compute_image_pair_similarity(
                        random_obs, obs
                    )
                    if score > best:
                        best = score
                best_scores.append(best)

            m = mean(best_scores) if len(best_scores) > 1 else 0.0
            s = stdev(best_scores) if len(best_scores) > 1 else 0.0
            print(f"mean: {m}, std: {s}")
            results[sc][traj_name] = {"mean": m, "std": s}

        # release gpu memory
        """try:
            sim.close()
        except Exception as e:
            print(e)"""

    return results

if __name__ == "__main__":
    #plot_trajectories() # comment out once you've generated plots once
    #plot_entropy_bars("results/traces.json") # comment out once you've generated plots once
    #plot_match_vs_entropy("results/match_scores.json", "results/sketch_entropies.json", save_path="results/match_vs_entropy.png") # comment out once you've generated plots once

    # code to run the experiments
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2"}:
        print("please use like: python experiments.py <1|2>")
        sys.exit(1)

    option = sys.argv[1]

    if option == "1":
        out, traces = build_sketches()
        out_file = "results/sketch_entropies.json"

        with open("results/traces.json", "w") as fp:
            json.dump(traces, fp, indent=2)

    if option == "2":
        out = compute_match_scores()
        out_file = "results/match_scores.json"

    with open(out_file, "w") as fp:
        json.dump(out, fp, indent=2)

    print(f"written to: {out_file}")

