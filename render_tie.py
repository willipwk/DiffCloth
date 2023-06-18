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
    perturbed_obj_files = glob.glob("output/025/episode1/results185/test1_*.obj")
    cpu_per_proc = 2
    total_cpu = os.cpu_count()
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
