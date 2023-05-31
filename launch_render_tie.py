import glob
import os
import sys
import time
from threading import Thread

import numpy as np
import open3d as o3d
import pybullet as p
from tqdm import tqdm
from tulip.utils.gl_utils import build_projection_matrix
from tulip.utils.image_utils import depth2xyz
from tulip.utils.pblt_utils import (
    build_view_matrix_pblt,
    get_vertices_pos,
    init_sim,
    render,
    vis_points,
)


def single2double_sided(in_obj_fn, out_obj_fn):
    w_fp = open(out_obj_fn, "w")
    faces = []
    with open(in_obj_fn) as r_fp:
        for line in r_fp:
            if line.startswith("v") or line.startswith("f"):
                w_fp.write(line)
            w = line.strip().split(" ")
            if w[0] == "f":
                faces.append(w[1:4])
    for f in faces:
        w_fp.write(
            "\nf {} {} {}".format(f[2], f[1], f[0])
        )  # add mesh in reverse side
    w_fp.close()


def prepare_for_multiproc(n_thread, urdf_fn):
    assert urdf_fn.endswith(".urdf"), "Invalid urdf filename."
    urdf_files = []
    for i in range(n_thread):
        if os.path.isdir(f"render_tmp_{i}"):
            os.system(f"rm -rf render_tmp_{i}")
        os.mkdir(f"render_tmp_{i}")
        os.system(f"cp {urdf_fn} render_tmp_{i}")
        tmp_urdf_fn = f"render_tmp_{i}/{urdf_fn.split('/')[-1]}"
        assert os.path.isfile(tmp_urdf_fn), "URDF copy failed."
        urdf_files.append(tmp_urdf_fn)
    return urdf_files


def render_thread(obj_fn, i, urdf_fn, mode="GUI"):
    # generate double-sided tie obj from perturbed tie for urdf
    perturbed_obj_fn = "output/" + "_".join(obj_fn.split("/")[-3:]).replace(
        ".obj", f"_perturbed_{i}.obj"
    )

    out_fn = perturbed_obj_fn.replace(".obj", "")
    if os.path.isfile(f"{out_fn}.ply"):
        return

    start_time = time.time()
    while not os.path.isfile(perturbed_obj_fn):
        with open("executed_files.txt", "r") as fp:
            executed_files = [l.strip() for l in fp.readlines()]
            if perturbed_obj_fn.replace("output/", "") in executed_files:
                with open("missing_files.txt", "a") as fp:
                    fp.write(f"{perturbed_obj_fn}\n")
                return
            else:
                print(f"Waiting for {perturbed_obj_fn} to be generated...")
                time.sleep(3)

    urdf_dir = "/".join(urdf_fn.split("/")[:-1])
    out_obj_fn = f"{urdf_dir}/tie.obj"
    single2double_sided(in_obj_fn=perturbed_obj_fn, out_obj_fn=out_obj_fn)

    # initialize simulation
    sim_cid = init_sim(mode=mode)
    tie_id = p.loadURDF(
        urdf_fn,
        basePosition=[0, 0, 0.05],
        baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, -np.pi / 2]),
        useFixedBase=True,
        physicsClientId=sim_cid,
    )

    # camera extrinsic related
    camera_pos = [-0.58451435, 0.0, 0.61609813]
    camera_quat = [
        -0.6733081448310533,
        0.6659691939501913,
        -0.22584407782434218,
        0.22833227394560413,
    ]
    view_matrix = build_view_matrix_pblt(
        camera_pos, camera_quat, sim_cid, vis=True
    )

    # camera intrinsic related
    width = 1920
    height = 1080
    fx = 1074.9383544900666
    fy = 1078.6895323593005
    cx = 954.0125249569526
    cy = 542.8760188199577
    far = 10
    near = 0.01
    proj_matrix = build_projection_matrix(
        width, height, fx, fy, cx, cy, near, far
    )

    # render image and get pointcloud
    rgb, depth, seg = render(
        width, height, view_matrix, proj_matrix, near, far, sim_cid
    )
    xyz = depth2xyz(
        depth,
        fx,
        fy,
        cx,
        cy,
        camera_pos,
        camera_quat,
        return_pcd=True,
        mask=(seg == seg.max()),
    )

    # save pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    n_points = len(pcd.points)
    downsample_size = 4096
    every_k_points = int(np.floor(n_points / downsample_size))
    dsmp_pcd = pcd.uniform_down_sample(every_k_points)
    o3d.io.write_point_cloud(f"{out_fn}.ply", dsmp_pcd)

    # save keypoint position
    scale = [0.15, 0.15, 0.15]  # hard-coded for now
    kp_indices = [8, 95, 182, 269, 356]
    v_pos = get_vertices_pos(tie_id, sim_cid, scale=scale)
    # vis_points(v_pos, sim_cid, color=[0, 1,0])
    with open(f"{out_fn}_kp_pos.npy", "wb") as fp:
        keypoints = [v_pos[idx] for idx in kp_indices]
        np.save(fp, np.array(keypoints))

    # check file
    if not (
        os.path.isfile(f"{out_fn}.ply")
        or os.path.isfile(f"{out_fn}_kp_pos.npy")
    ):
        print(f"[Warning]: failed to generate {out_fn}.ply")
    p.disconnect(sim_cid)


if __name__ == "__main__":

    obj_files = glob.glob(
        "src/assets/meshes/remeshed/tie_data/*/*/tie_final.obj"
    )
    obj_files = sorted(obj_files)

    n_output = 500
    cpu_per_proc = 2
    total_cpu = os.cpu_count()
    n_thread = int(total_cpu / cpu_per_proc)
    urdf_files = prepare_for_multiproc(n_thread, "tie/tie.urdf")
    for start_idx in range(0, len(obj_files), n_thread):
        for i in range(n_output):
            # threads = []
            for thread_idx in range(min(len(obj_files) - start_idx, n_thread)):
                obj_fn = obj_files[start_idx + thread_idx]
                try:
                    render_thread(obj_fn, i, urdf_files[thread_idx], "DIRECT")
                except:
                    perturbed_obj_fn = "output/" + "_".join(
                        obj_fn.split("/")[-3:]
                    ).replace(".obj", f"_perturbed_{i}.obj")
                    with open("missing_files.txt", "a") as fp:
                        fp.write(f"{perturbed_obj_fn}\n")

                # threads.append(
                #    Thread(
                #        target=render_thread,
                #        args=(obj_fn, i, urdf_files[thread_idx], "DIRECT"),
                #    )
                # )
            # for t in threads:
            #    t.start()
            # for t in threads:
            #    t.join()
            print("=" * 50, start_idx, i)
