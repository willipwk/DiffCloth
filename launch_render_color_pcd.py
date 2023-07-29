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
    input_root_dir = "./tmp_blender/026/episode3/results583/"
    key_root_path = "/026/episode3/results583/"
    preprocess_dir = os.listdir(input_root_dir)
    cpu_per_proc = 2
    total_cpu = os.cpu_count() // 2
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

    shutil.rmtree("./tmp_blender/")
    shutil.rmtree("./tmp_double/")
