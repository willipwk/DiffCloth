import glob
import os
from threading import Thread
from tqdm import tqdm
import time


def render_thread(perturb_fn: str, urdf_fn: str, obj_fn: str, render_mode: str):
    cmd = f"python launch_render_tie.py {perturb_fn} {urdf_fn} {obj_fn} {render_mode}"
    os.system(cmd)


if __name__ == "__main__":
    render_mode = "DIRECT"
    obj_files = [
        "output/action/025/episode1/results185/*.obj",
        # "output/025/episode2/results409/*.obj",
        # "output/025/episode3/results729/*.obj",
        # "output/025/episode4/results1073/*.obj",
        # "output/025/episode5/results1257/*.obj",
        # "output/025/episode6/results1577/*.obj",
        # "output/026/episode1/results097/*.obj",
        # "output/026/episode1/results175/*.obj",
        # "output/026/episode2/results301/*.obj",
        # "output/026/episode2/results421/*.obj",
        # "output/026/episode4/results919/*.obj",
        # "output/026/episode4/results1231/*.obj",
        # "output/026/episode5/results1531/*.obj",
        # "output/026/episode5/results1651/*.obj",
        # "output/026/episode6/results1891/*.obj",
        # "output/026/episode7/results2341/*.obj",
        # "output/026/episode4/results997/*.obj",
        # "output/026/episode4/results1177/*.obj",
        # "output/026/episode4/results1273/*.obj",
        # "output/026/episode4/results1339/*.obj",
        # "output/026/episode5/results1711/*.obj",
        # "output/026/episode6/results2185/*.obj",
        # "output/026/episode7/results2395/*.obj",
    ]
    perturbed_obj_files = []
    for obj in obj_files:
        perturbed_obj_files += glob.glob(obj)
    perturbed_obj_files.sort()
    cpu_per_proc = 2
    total_cpu = 16
    n_thread = int(total_cpu / cpu_per_proc)
    i = 0
    start_time = time.time()
    while i < len(perturbed_obj_files):
        iter_start_time = time.time()
        next_render_num = min(len(perturbed_obj_files) - i, n_thread)
        threads = []
        for thread_idx in range(next_render_num):
            threads.append(
                Thread(
                    target=render_thread,
                    args=(
                        perturbed_obj_files[i + thread_idx],
                        f"./tie/tie{thread_idx}.urdf",
                        f"./tie/tie{thread_idx}.obj",
                        render_mode
                    ),
                )
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        i += next_render_num
        end_time = time.time()
        print("iteration time usage", (end_time - iter_start_time), " total time usage", (end_time - start_time))
        print("=" * 50, i, len(perturbed_obj_files))
