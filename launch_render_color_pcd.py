import glob
import os
from threading import Thread
from tqdm import tqdm
import time
import shutil


def render_thread(input_dir: str, key_root_path: str, count: int):
    cmd = f"python render_color_pcd.py {input_dir} {key_root_path} {count}"
    os.system(cmd)


if __name__ == "__main__":
    render_mode = "DIRECT"
    obj_files = [
        "/025/episode1/results185/",
        # "/025/episode2/results409/",
        # "/025/episode3/results729/",
        # "/025/episode4/results1073/",
        # "/025/episode5/results1257/",
        # "/025/episode6/results1577/",
        # "/026/episode1/results097/",
        # "/026/episode1/results175/",
        # "/026/episode2/results301/",
        # "/026/episode2/results421/",
        # "/026/episode4/results919/",
        # "/026/episode4/results1231/",
        # "/026/episode5/results1531/",
        # "/026/episode5/results1651/",
        # "/026/episode6/results1891/",
        # "/026/episode7/results2341/",
    ]
    for obj in obj_files:
        input_root_dir = "./tmp_blender" + obj
        key_root_path = obj
        preprocess_dir = os.listdir(input_root_dir)
        cpu_per_proc = 2
        total_cpu = 32
        n_thread = int(total_cpu / cpu_per_proc)
        i = 0
        start_time = time.time()
        while i < len(preprocess_dir):
            iter_start_time = time.time()
            next_render_num = min(len(preprocess_dir) - i, n_thread)
            threads = []
            for thread_idx in range(next_render_num):
                threads.append(
                    Thread(
                        target=render_thread,
                        args=(
                            input_root_dir + preprocess_dir[i + thread_idx],
                            key_root_path,
                            i + thread_idx,
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
            print("=" * 50, i, len(preprocess_dir))

        # shutil.rmtree("./tmp_blender/025/episode1/results185")
        # shutil.rmtree("./tmp_double/025/episode1/results185")
