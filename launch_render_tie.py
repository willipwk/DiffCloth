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
import torch
from scipy.spatial.transform import Rotation
import pygeodesic.geodesic as geodesic
import open3d as o3d
import pybullet as p
from tqdm import tqdm
import point_cloud_utils as pcu
from tulip.utils.gl_utils import build_projection_matrix
from tulip.utils.image_utils import depth2xyz
from tulip.utils.pblt_utils import (
    build_view_matrix_pblt,
    get_vertices_pos,
    init_sim,
    render,
    vis_points,
)


def read_obj(obj_file_path: str):
    """
    read obj file to generate vertices and faces
    """
    with open(obj_file_path) as file:
        points = []
        faces = []
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append([float(strs[1]), float(strs[2]), float(strs[3])])
            if strs[0] == "vt":
                continue
            if strs[0] == "f":
                faces.append([int(strs[1]) - 1, int(strs[2]) - 1, int(strs[3]) - 1])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces


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


def cal_normal(vertices, faces, pcl):
    n = pcu.estimate_mesh_face_normals(vertices, faces)
    d, fi, bc = pcu.closest_points_on_mesh(pcl, vertices, faces)
    normal = n[fi]
    return normal


def cal_midline(vertices, pcl):
    normal_midline = vertices[1:358] - vertices[7:364]
    endline_midline1 = vertices[1:4] - vertices[4:7]
    endline_midline2 = vertices[358:361] - vertices[361:364]
    endpoint_midline1 = vertices[[0]] - vertices[[2]]
    endpoint_midline2 = vertices[[362]] - vertices[[364]]
    midline = np.vstack([endpoint_midline1, endline_midline1, normal_midline, endline_midline2, endpoint_midline2])
    midline = midline / np.linalg.norm(midline, axis=1, keepdims=True)
    pcl_torch = torch.from_numpy(pcl)
    vertices_torch = torch.from_numpy(vertices)
    shortest_dist = torch.cdist(pcl_torch, vertices_torch)
    pcd_assign = torch.argmin(shortest_dist, dim=1).tolist()
    pcl_midline = midline[pcd_assign]
    return pcl_midline


def render_thread(perturbed_obj_fn: str, urdf_fn: str, render_obj_fn: str, mode="GUI"):

    rnd = np.random.uniform()
    if rnd < 0.8:
        data_mode = "train"
    elif rnd < 0.9:
        data_mode = "eval"
    else:
        data_mode = "test"
    perturbed_obj_folder = perturbed_obj_fn.replace("output/action/", "").replace(".obj", "")
    data_path = "../pointnet2_keypoint_prediction/datasets/action/" + perturbed_obj_folder[:perturbed_obj_folder.rfind("/")] + "/" + data_mode
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

    # # camera extrinsic related kinect
    # camera_pos = [-0.58451435, 0.0, 0.61609813]
    # camera_quat = [
    #     -0.6733081448310533,
    #     0.6659691939501913,
    #     -0.22584407782434218,
    #     0.22833227394560413,
    # ]
    # camera extrinsic related realsense
    camera_pos = [0.0, 0.0, 0.597076]
    r = Rotation.from_euler(seq='xyz', angles=[np.pi, 0, -np.pi / 2])
    camera_quat = Rotation.as_quat(r)
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
    vis_points(xyz, sim_cid, color=[0, 1, 0])

    scale = [1.0, 1.0, 1.0]
    v_pos = get_vertices_pos(tie_id, sim_cid, scale=scale)
    v_pos = np.array(v_pos)
    # save pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    n_points = len(pcd.points)
    downsample_size = 4096
    every_k_points = int(np.floor(n_points / downsample_size))
    dsmp_pcd = pcd.uniform_down_sample(every_k_points)
    # dsmp_xyz = np.array(dsmp_pcd.points)
    # r_array = np.random.uniform(140 / 255, 160 / 255, (dsmp_xyz.shape[0], 1))
    # g_array = np.random.uniform(30 / 255, 50 / 255, (dsmp_xyz.shape[0], 1))
    # b_array = np.random.uniform(35 / 255, 60 / 255, (dsmp_xyz.shape[0], 1))
    # rgb = np.hstack([r_array, g_array, b_array])
    # dsmp_pcd.colors = o3d.utility.Vector3dVector(rgb)
    # _, faces = pcu.load_mesh_vf(perturbed_obj_fn)
    # normal = cal_normal(v_pos, faces, dsmp_xyz)
    # dsmp_pcd.normals = o3d.utility.Vector3dVector(normal)
    # dsmp_pcd.paint_uniform_color([0, 1, 0])
    o3d.io.write_point_cloud(f"{out_fn}.ply", dsmp_pcd)

    dsmp_points = np.array(dsmp_pcd.points)
    # save keypoint position
    # scale = [0.15, 0.15, 0.15]  # hard-coded for now
    kp_indices = [8, 95, 182, 269, 356]
    # compute geodesic distance
    points, faces = read_obj(perturbed_obj_fn)
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    source_indices = np.array(kp_indices) 
    target_indices = None
    geodesic_list = []
    for kp in kp_indices: 
        source_indices = np.array([kp])
        distances, best_source = geoalg.geodesicDistances(source_indices, target_indices)
        geodesic_list.append(distances)
    vertex_geodesic_dist = np.stack(geodesic_list, axis=-1)
    # np.save(f"{out_fn}_geodesic_dist.npy", vertex_geodesic_dist)
    # pcl_midline = cal_midline(v_pos, dsmp_points)
    # np.save(f"{out_fn}_midline.npy", pcl_midline)
    # vis_points(v_pos, sim_cid, color=[0, 1, 0])
    np.save(f"{out_fn}_v_pos.npy", v_pos)
    # with open(f"{out_fn}_kp_pos.npy", "wb") as fp:
    #     np.save(fp, np.array(keypoints))
    # for i, pos in enumerate(keypoints):
    #     print(
    #         i,
    #         np.linalg.norm(xyz - pos, axis=-1).min(),
    #         np.linalg.norm(np.array(pcd.points) - pos, axis=-1).min(),
    #         np.linalg.norm(np.array(dsmp_pcd.points) - pos, axis=-1).min(),
    #     )
    # o3d.visualization.draw_geometries([dsmp_pcd])
    # time.sleep(100000)
    anchor_vertex = 182
    start_vertex = 356
    anchor_distance = np.loadtxt("ep1_displace.txt")
    start_distance = v_pos[anchor_vertex] - v_pos[start_vertex]
    final_distance = start_distance + anchor_distance
    np.save(f"{out_fn}_displace.npy", final_distance)

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
