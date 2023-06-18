import glob
import os
import sys
import time
from threading import Thread
import math
import shutil
import xml.dom.minidom
import numpy as np
import scipy
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
    if os.path.isfile(out_obj_fn):
        os.system(f"rm {out_obj_fn}")
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


def prepare_urdf(urdf_fn: str, obj_fn: str):
    # copy template
    shutil.copyfile("./tie/tie.urdf", urdf_fn)
    # change mesh file path
    tie_urdf = xml.dom.minidom.parse(urdf_fn)
    mesh_file_list = tie_urdf.getElementsByTagName("mesh")
    for mesh_file in mesh_file_list:
        mesh_file.setAttribute("filename", obj_fn)
    with open(urdf_fn, "w") as f:
        tie_urdf.writexml(f)


def render_thread(perturbed_obj_fn: str, urdf_fn: str, render_obj_fn: str, mode="GUI"):

    rnd = np.random.uniform()
    if rnd < 0.8:
        data_mode = "train"
    elif rnd < 0.9:
        data_mode = "eval"
    else:
        data_mode = "test"
    perturbed_obj_folder = perturbed_obj_fn.replace("output/", "").replace(".obj", "")
    data_path = "../pointnet2_keypoint_prediction/datasets/" + perturbed_obj_folder[:perturbed_obj_folder.rfind("/")] + "/" + data_mode
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    out_fn = data_path + perturbed_obj_fn[perturbed_obj_fn.rfind("/"):].replace(".obj", "")

    if not os.path.isfile(urdf_fn):
        prepare_urdf(urdf_fn, render_obj_fn)

    single2double_sided(in_obj_fn=perturbed_obj_fn, out_obj_fn=render_obj_fn)

    # initialize simulation
    sim_cid = init_sim(mode=mode)
    print("thread num:", render_obj_fn[-5], "sim cid:", sim_cid)
    tie_id = p.loadURDF(
        urdf_fn,
        basePosition=[0, 0, 0.0],
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
    np.random.shuffle(xyz)
    # vis_points(xyz, sim_cid, color=[0, 1, 0])

    # save pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    n_points = len(pcd.points)
    downsample_size = 4096
    every_k_points = int(np.floor(n_points / downsample_size))
    dsmp_pcd = pcd.uniform_down_sample(every_k_points)
    # dsmp_pcd.paint_uniform_color([0, 1, 0])
    o3d.io.write_point_cloud(f"{out_fn}.ply", dsmp_pcd)

    # save keypoint position
    # scale = [0.15, 0.15, 0.15]  # hard-coded for now
    scale = [1.0, 1.0, 1.0]
    kp_indices = [8, 95, 182, 269, 356]
    v_pos = get_vertices_pos(tie_id, sim_cid, scale=scale)
    # vis_points(v_pos, sim_cid, color=[0, 1, 0])
    keypoints = [v_pos[idx] for idx in kp_indices]
    np.save(f"{out_fn}_kp_pos.npy", np.array(keypoints))
    # with open(f"{out_fn}_kp_pos.npy", "wb") as fp:
    #     np.save(fp, np.array(keypoints))
    # for i, pos in enumerate(keypoints):
    #     print(
    #         i,
    #         np.linalg.norm(xyz - pos, axis=-1).min(),
    #         np.linalg.norm(np.array(pcd.points) - pos, axis=-1).min(),
    #         np.linalg.norm(np.array(dsmp_pcd.points) - pos, axis=-1).min(),
    #     )
    # kp = o3d.geometry.PointCloud()
    # kp.points = o3d.utility.Vector3dVector(np.array(keypoints))
    # kp.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([dsmp_pcd] + [kp])
    # time.sleep(100000)

    # check file
    if not (
        os.path.isfile(f"{out_fn}.ply")
        or os.path.isfile(f"{out_fn}_kp_pos.npy")
    ):
        print(f"[Warning]: failed to generate {out_fn}.ply")
    p.disconnect(sim_cid)


if __name__ == "__main__":
    # perturbed_obj_files = glob.glob("output/025/episode2/results361/test2_*.obj")
    # for fn in tqdm(perturbed_obj_files):
    render_thread(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # prepare_urdf("./tie/tie1.urdf", "./tie/tie1.obj")
