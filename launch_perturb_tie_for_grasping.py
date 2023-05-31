import glob
import os
import sys
import time
import numpy as np
from threading import Thread


def perturb_thread(obj_fn, i, seed, n_openmp_thread):

    in_fn = obj_fn[obj_fn.find("remeshed") :]

    vertex_fn = obj_fn.replace("grasping_obj/grasp", "grasping_obj/vertex")
    vertex_fn = vertex_fn.replace(".obj", ".txt")
    att_idx = np.loadtxt(vertex_fn).tolist()
    att_idx = [int(idx) for idx in att_idx if idx != -1]
    att_idx = np.random.choice(att_idx)

    out_fn = "_".join(obj_fn.split("/")[-3:]).replace(
        ".obj", f"_perturbed_{i}.obj"
    )
    cmd = "python3 src/python_code/perturb_tie.py"
    cmd += " -s"
    cmd += " -mode 0"
    cmd += f" -att_idx {att_idx}"
    cmd += " -task_name perturb_tie"
    cmd += f" -in_fn {in_fn}"
    cmd += f" -out_fn {out_fn}"
    cmd += f" -n_openmp_thread {n_openmp_thread}"
    cmd += f" -seed {seed + i}"

    #print(cmd)
    os.system(cmd)

    if not os.path.isfile(f"output/{out_fn}"):
        print(f"[Warning]: failed to generate {out_fn}")
    else:
        os.system(f"rm output/{out_fn.replace('.obj', '.txt')}")


if __name__ == "__main__":

    obj_files = glob.glob(
        "src/assets/meshes/remeshed/grasping_obj/grasp*.obj"
    )
    obj_files = sorted(obj_files)

    n_output = 500
    cpu_per_proc = 3
    total_cpu = os.cpu_count()
    n_thread = int(total_cpu / cpu_per_proc)
    executed_files = []
    for start_idx in range(0, len(obj_files), n_thread):
        for i in range(n_output):
            threads = []
            for thread_idx in range(min(len(obj_files) - start_idx, n_thread)):
                obj_fn = obj_files[start_idx + thread_idx]
                seed = (start_idx + thread_idx) * 123456
                #perturb_thread(obj_fn, i, seed, cpu_per_proc)
                threads.append(
                    Thread(
                        target=perturb_thread,
                        args=(
                            obj_fn,
                            i,
                            seed,
                            cpu_per_proc,
                        ),
                    )
                )
                perturb_fn = "_".join(obj_fn.split("/")[-3:]).replace(
                    ".obj", f"_perturbed_{i}.obj"
                )
                executed_files.append(perturb_fn)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            with open("executed_files.txt", "w") as fp:
                fp.write("\n".join(executed_files) + "\n")
            print("=" * 50, start_idx, i)
