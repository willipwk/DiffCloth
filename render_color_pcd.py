import time
import open3d as o3d
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from tulip.utils.gl_utils import build_projection_matrix, build_view_matrix
from tulip.utils.image_utils import vis_depth, vis_rgb, vis_seg_indices, depth2xyz
from tulip.utils.pblt_utils import (
    build_view_matrix_pblt,
    init_sim,
    render,
    vis_frame,
    vis_points,
    get_vertices_pos,
)
from typing import List, Tuple
from scipy.spatial.transform import Rotation
from tulip.utils.transform_utils import homogeneous_transform
from tulip.utils.transform_utils import pos_quat2pose_matrix, homogeneous_transform
# from file_utils import *
import matplotlib.pyplot as plt
# import cv2
# import torch
import os
import shutil
# import albumentations as A
# import h5py
import point_cloud_utils as pcu
import random
import sys


def gen_urdf(urdf_filename, robotname, mesh_filename, scale, colorname, rgba):
    f = open(urdf_filename, 'w')
    f.write('<?xml version="1.0" ?>\n')
    f.write('<robot name="{}">\n'.format(robotname))
    f.write('<link name="Base">\n')
    f.write('    <contact>\n')
    f.write('    <lateral_friction value="1.0"/>\n')
    f.write('    <rolling_friction value="0.0"/>\n')
    f.write('    <contact_cfm value="0.0"/>\n')
    f.write('    <contact_erp value="1.0"/>\n')
    f.write('    </contact>\n')
    f.write('    <inertial>\n')
    f.write('    <origin rpy="0 0 0" xyz="0 0 0"/>\n')
    f.write('    <mass value="1.0"/>\n')
    f.write('    <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>\n')
    f.write('    </inertial>\n')
    f.write('    <visual>\n')
    f.write('    <origin rpy="0 0 0" xyz="0 0 0"/>\n')
    f.write('    <geometry>\n')
    f.write('        <mesh filename="{}" scale="{} {} {}"/>\n'.format(mesh_filename, scale, scale, scale))
    f.write('    </geometry>\n')
    f.write('    <material name="{}">\n'.format(colorname))
    f.write('        <color rgba="{} {} {} {}"/>\n'.format(rgba[0], rgba[1], rgba[2], rgba[3]))
    f.write('    </material>\n')
    f.write('    </visual>\n')
    f.write('    <collision>\n')
    f.write('    <origin rpy="0 0 0" xyz="0 0 0"/>\n')
    f.write('    <geometry>\n')
    f.write('        <mesh filename="{}" scale="{} {} {}"/>\n'.format(mesh_filename, scale, scale, scale))
    f.write('    </geometry>\n')
    f.write('    </collision>\n')
    f.write('</link>\n')
    f.write('</robot>\n')
    f.close()


# def load_obj_only(source_filename):     # only load obj, not convert
#     f = open(source_filename, 'r')
#     verts = []
#     faces = []
#     for line in f:
#         if line[-1] == '\n':
#             line = line[:-1]
#         wlist = line.split(' ')
#         if wlist[0] == 'f':
#             faces.append(wlist[1:4])
#         elif wlist[0] == 'v':
#             verts.append(wlist[1:4])
#     f.close()

#     verts = np.array([tuple(map(float, v)) for v in verts])
#     faces = np.array([tuple(map(int, map(eval, f))) for f in faces])

#     return verts, faces


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


def depth2xyz_mat(depth, fx, fy, cx, cy, cam_pos, cam_quat):
    print("==> Converting depth to xyz in matrix version")
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    h, w = depth.shape
    pcd = []
    xyz_image = np.zeros((h, w, 3), np.float32)
    
    h_mat = np.tile(np.array(range(h)).reshape(-1,1), (1, w))
    w_mat = np.tile(np.array(range(w)), (h, 1))

    z = depth
    pos_vec = np.array([(w_mat - cx) * z / fx, (h_mat - cy) * z / fy, z, np.ones((h, w))])

    xyz_image = np.matmul(cam_pose, pos_vec.transpose(1, 0, 2))[:, :3, :]
    xyz_image = xyz_image.transpose(0, 2, 1)

    return xyz_image, pcd    # no pcd here


def pcd2xyz(pcd, width, height, fx, fy, cx, cy, cam_pos, cam_quat):
    print("==> Converting pcd to xyz image")
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    cam_extrinsic = np.linalg.inv(cam_pose)
    xyz_image = np.zeros((height, width, 3), np.float32)
    for point in pcd:
        pos_vec = np.array([point[0], point[1], point[2], 1])
        uvz_vec = np.matmul(cam_extrinsic, pos_vec.transpose())[:3]
        z = uvz_vec[2]
        w_i = int(uvz_vec[0] * fx / z + cx)
        h_i = int(uvz_vec[1] * fy / z + cy)
        xyz_image[h_i, w_i] = pos_vec[:3]
    return xyz_image


def pcd2uv(pcd, width, height, fx, fy, cx, cy, cam_pos, cam_quat):
    print("==> Converting pcd to uv image")
    cam_pose = pos_quat2pose_matrix(cam_pos, cam_quat)
    cam_extrinsic = np.linalg.inv(cam_pose)
    uv_image = np.zeros((len(pcd), 2), np.int32)
    for point_id, point in enumerate(pcd):
        pos_vec = np.array([point[0], point[1], point[2], 1])
        uvz_vec = np.matmul(cam_extrinsic, pos_vec.transpose())[:3]
        z = uvz_vec[2]
        u_i = int(uvz_vec[0] * fx / z + cx)
        v_i = int(uvz_vec[1] * fy / z + cy)
        uv_image[point_id] = np.array([u_i, v_i])
    return uv_image


def cal_normal(vertices, faces, pcl):
    n = pcu.estimate_mesh_face_normals(vertices, faces)
    d, fi, bc = pcu.closest_points_on_mesh(pcl.reshape(pcl.shape[0] * pcl.shape[1], pcl.shape[2]), vertices, faces)
    normal = n[fi]
    return normal

def render_one_tie( input_dir,
                    filename,
                    output_dir, 
                    front_texure_path, 
                    back_texture_path, 
                    tie_pos, 
                    tie_ori, 
                    tie_scale, 
                    tie_color1, 
                    tie_color2, 
                    cam_pos, 
                    cam_quat,
                    result_id):    # filename: .obj

    # initialize simulation
    # mode = "GUI"
    mode = "DIRECT"
    sim_cid = init_sim(mode=mode)
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    
    input_path = os.path.join(input_dir, filename)
    # every obj is a dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    obj_front_side_path = os.path.join(input_dir, 'front.obj')
    obj_back_side_path = os.path.join(input_dir, 'back.obj')

    # v_local_pos, faces = objloader(obj_front_side_path, obj_back_side_path)
    # print(vertices)

    v_local_pos, faces = pcu.load_mesh_vf(obj_front_side_path)
    # print(faces)
    # points, faces = read_obj(obj_back_side_path)

    # desk_urdf_path = os.path.join(output_dir, "desk.urdf")
    # gen_urdf(desk_urdf_path, 'desk', 'desk.obj', '1.0', 'white', desk_color)

    obj_urdf_path1 = os.path.join(input_dir, 'tie_front.urdf') 
    gen_urdf(obj_urdf_path1, 'tie_front', obj_front_side_path, tie_scale, 'blue', tie_color1)

    obj_urdf_path2 = os.path.join(input_dir, 'tie_back.urdf') 
    gen_urdf(obj_urdf_path2, 'tie_back', obj_back_side_path, tie_scale, 'gray', tie_color2)


    # desk_id = p.loadURDF(
    #     desk_urdf_path,
    #     basePosition=[0, 0, 0],
    #     baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]),
    #     useFixedBase=True,
    #     physicsClientId=sim_cid,
    # )

    tie_front_id = p.loadURDF(
        obj_urdf_path1,
        basePosition=tie_pos,
        baseOrientation=p.getQuaternionFromEuler(tie_ori),
        useFixedBase=True,
        physicsClientId=sim_cid,
    )

    tie_back_id = p.loadURDF(
        obj_urdf_path2,
        basePosition=tie_pos,
        baseOrientation=p.getQuaternionFromEuler(tie_ori),
        useFixedBase=True,
        physicsClientId=sim_cid,
    )
    
    # p.setGravity(0, 0, -9.8)
    for _ in range(20):
        p.stepSimulation()
        if mode=='GUI':
            time.sleep(1/240.)

    front_texture_id = p.loadTexture(front_texture_path)
    p.changeVisualShape(tie_front_id, -1, textureUniqueId=front_texture_id)
    back_texture_id = p.loadTexture(back_texture_path)
    p.changeVisualShape(tie_back_id, -1, textureUniqueId=back_texture_id)

    tie_vs = p.getVisualShapeData(tie_front_id, sim_cid)
    tie_pos, tie_quat = p.getBasePositionAndOrientation(tie_front_id, sim_cid)
    tie_scale = np.array(tie_vs[0][3])

    # v_pos = []
    # for pos in v_local_pos:
    #     v_pos.append(
    #         homogeneous_transform(
    #             tie_pos, tie_quat, pos * tie_scale, [0, 0, 0, 1]
    #         )[0]
    #     )
    v_pos = get_vertices_pos(tie_front_id, sim_cid)
    vertices = np.array(v_pos)
    # np.savetxt("vertices.xyz", vertices)

    p.changeVisualShape(tie_front_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED) 
    p.changeVisualShape(tie_back_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED) 

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    keypoint_id = [14, 98, 182, 266, 350]
    keypoint_groups = [[14, 10, 11, 12, 13, 15, 17],
                [98, 94, 95, 96, 97, 99, 101],
                [182, 178, 179, 180, 181, 183, 185],
                [266, 262, 263, 264, 265, 267, 269],
                [350, 346, 347, 348, 349, 351, 353]]

    # extrinsic related
    camera_pos = cam_pos
    camera_quat = cam_quat
    vis_frame(camera_pos, camera_quat, sim_cid, length=0.2, duration=15)
    gl_view_matrix = build_view_matrix(camera_pos, camera_quat)
    pblt_view_matrix = build_view_matrix_pblt(
        camera_pos, camera_quat, sim_cid, vis=True
    )

    # camera extrinsic related realsense
    camera_pos = [0.0, 0.0, 0.797076]
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

    # render and visualize
    # for view_matrix in [gl_view_matrix, pblt_view_matrix]:
    # for view_matrix in [pblt_view_matrix]:
    rgb, depth, seg = render(
        width, height, view_matrix, proj_matrix, near, far, sim_cid
    )

    vis_rgb = ((rgb / float(255)) * 255.0).astype(np.uint8)

    # vis_depth = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)

    xyz_image, pcd = depth2xyz_mat(
        depth, fx, fy, cx, cy, camera_pos, camera_quat
    )
    # xyz_image2 = depth2xyz(depth, fx, fy, cx, cy, camera_pos, camera_quat)
    # print(xyz_image1.shape)
    # print(xyz_image2.shape)

    mask = np.where(seg < 0, 0, 1)
    mask = np.reshape(mask, (mask.shape[0] * mask.shape[1],))
    valid_index = np.nonzero(mask)

    # xyz_image2 = np.reshape(xyz_image2, (xyz_image2.shape[0] * xyz_image2.shape[1], xyz_image2.shape[2]))
    # seg_xyz = xyz_image2[valid_index]
    # np.savetxt("seg_xyz2.xyz", seg_xyz)

    # uv_image = pcd2uv(
    #     vertices, width, height, fx, fy, cx, cy, camera_pos, camera_quat
    # )

    normal = cal_normal(vertices, faces, xyz_image)
    normal = np.reshape(normal, (xyz_image.shape[0], xyz_image.shape[1], xyz_image.shape[2]))
    full_pcd = np.dstack([xyz_image, vis_rgb / 255, normal])
    full_pcd = np.reshape(full_pcd, (full_pcd.shape[0] * full_pcd.shape[1], full_pcd.shape[2]))
    full_pcd = full_pcd[valid_index]
    np.random.shuffle(full_pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_pcd[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(full_pcd[:, 3:6])
    pcd.normals = o3d.utility.Vector3dVector(full_pcd[:, 6:])
    n_points = len(pcd.points)
    downsample_size = 4096
    every_k_points = int(np.floor(n_points / downsample_size))
    dsmp_pcd = pcd.uniform_down_sample(every_k_points)
    # o3d.visualization.draw_geometries([dsmp_pcd], window_name="example", point_show_normal=True)
    # dsmp_pcd.paint_uniform_color([0, 1, 0])
    o3d.io.write_point_cloud(f"{output_dir}result{result_id}.ply", dsmp_pcd)
    # o3d.io.write_point_cloud(f"result.ply", dsmp_pcd)

    if mode == 'GUI':
        time.sleep(5)
        input("enter to continue")
    p.disconnect(sim_cid)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    key_root_path = sys.argv[2]
    count = int(sys.argv[3])

    filename = "front.obj"
    rnd = np.random.uniform()
    if rnd < 0.8:
        data_mode = "train"
    elif rnd < 0.9:
        data_mode = "eval"
    else:
        data_mode = "test"
    output_dir = "../pointnet2_keypoint_prediction/datasets/normal/" + key_root_path + "/" + data_mode + "/"
    front_texture_path =  "./big_108.png"
    back_texture_path = "./big_107.png"

    tie_color1 = [100 / 255, 151 / 255, 202 / 255, 1]   # texture会叠加原来的颜色
    tie_color2 = [255 / 255, 255 / 255, 255 / 255, 1]
    desk_color = [242 / 255, 242 / 255, 242 / 255, 1]

    tie_pos = [0, 0, 0]
    tie_ori = [np.pi / 2, 0, np.pi / 2]
    tie_scale = 0.25

    cam_pos = [-0.1, 0.1, 0.8]
    cam_quat = p.getQuaternionFromEuler([-2.9877652, -0.0000724, -1.5598398 ])   # rad  /  YZX
    print(cam_quat)
    render_one_tie(input_dir, 
                    filename, 
                    output_dir, 
                    front_texture_path, 
                    back_texture_path, 
                    tie_pos, 
                    tie_ori, 
                    tie_scale, 
                    tie_color1, 
                    tie_color2, 
                    cam_pos, 
                    cam_quat,
                    count)