import glob
import os
import sys
import time
from threading import Thread


def perturb_thread(obj_fn: str, i: int, seed: int, n_openmp_thread: int):

    in_fn = obj_fn[obj_fn.find("remeshed") :]

    # out_fn = "_".join(obj_fn.split("/")[-4:]).replace(
    #     ".obj", f"_perturbed_{i}.obj"
    # )
    out_fn = in_fn[9:].replace(
        ".obj", f"_perturbed_{i}.obj"
    )
    cmd = "python3 src/python_code/perturb_tie.py"
    cmd += " -s"
    # cmd += " -r"
    cmd += " -mode 2"
    cmd += " -task_name perturb_tie"
    cmd += f" -in_fn {in_fn}"
    cmd += f" -out_fn {out_fn}"
    cmd += f" -n_openmp_thread {n_openmp_thread}"
    cmd += f" -seed {seed + i}"

    # print(cmd)
    os.system(cmd)

    if not os.path.isfile(f"output/action/{out_fn}"):
        print(f"[Warning]: failed to generate {out_fn}")
    else:
        os.system(f"rm output/action/{out_fn.replace('.obj', '.txt')}")


if __name__ == "__main__":

    obj_files = [
        "src/assets/meshes/remeshed/025/episode1/results185/tie_final.obj",
        # "src/assets/meshes/remeshed/025/episode2/results409/tie_final.obj",
        # "src/assets/meshes/remeshed/025/episode3/results729/tie_final.obj",
        # "src/assets/meshes/remeshed/025/episode4/results1073/tie_final.obj",
        # "src/assets/meshes/remeshed/025/episode5/results1257/tie_final.obj",
        # "src/assets/meshes/remeshed/025/episode6/results1577/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode4/results997/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode4/results1177/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode4/results1273/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode4/results1339/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode5/results1711/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode6/results2185/tie_final.obj",
        # "src/assets/meshes/remeshed/026/episode7/results2395/tie_final.obj",
    ]
    # obj_files += glob.glob(
    #   "src/assets/meshes/remeshed/tie_data/episode5/*/*.obj"
    # )
    obj_files = sorted(obj_files)

    n_output = 500
    cpu_per_proc = 2
    total_cpu = 16
    n_thread = int(total_cpu / cpu_per_proc)
    executed_files = []
    for start_idx in range(len(obj_files)):
        i = 0
        obj_start_idx = 0
        obj_fn = obj_files[start_idx]
        out_folder = "output/action" + obj_fn[obj_fn.find("remeshed") + len("remeshed"):obj_fn.rfind("/") + 1]
        while i < n_output:
            threads = []
            for thread_idx in range(min(n_output - i, n_thread)):
                seed = (start_idx + obj_start_idx + thread_idx) * 123456
                print("begin:", obj_start_idx + thread_idx)
                threads.append(
                    Thread(
                        target=perturb_thread,
                        args=(
                            obj_fn,
                            obj_start_idx + thread_idx,
                            seed,
                            cpu_per_proc,
                        ),
                    )
                )
                # perturb_fn = "_".join(obj_fn.split("/")[-3:]).replace(
                #     ".obj", f"_perturbed_{i}.obj"
                # )
                # perturb_fn = obj_fn[obj_fn.find("remeshed") + len("remeshed"):].replace(
                #     ".obj", f"_perturbed_{i}.obj"
                # )
                # executed_files.append(perturb_fn)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            obj_start_idx += min(n_output - i, n_thread)
            perturbed_obj = glob.glob(
                                "output/action" + obj_fn[obj_fn.find("remeshed") + len("remeshed"):obj_fn.rfind("/") + 1] + "*.obj"
                            )
            i = len(perturbed_obj)
            print("=" * 50, start_idx, i)

        perturbed_obj = os.listdir(out_folder)
        with open("executed_files.txt", "w") as fp:
            fp.write("\n".join(perturbed_obj) + "\n")
