#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : demo.py
#   Author      : YunYang1994
#   Created date: 2020-09-03 21:52:12
#   Description :
#
# ================================================================

import numpy as np

import h5py
import smplx
import torch
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R

sample = h5py.File("data/sample.h5")
model  = smplx.create(model_path=".",
                      model_type="smplx",
                      gender="male",
                      use_face_contour=True,
                      num_betas=10,
                      num_expression_coeffs=10)
model.use_pca = False

betas      = torch.from_numpy(sample['betas'].value)
body_pose  = sample['body_pose'].value

left_hand_pose  = torch.from_numpy(sample['left_hand_pose'].value)
right_hand_pose = torch.from_numpy(sample['right_hand_pose'].value)

body_pose       = R.from_matrix(body_pose).as_rotvec().reshape(1, -1)
left_hand_pose  = R.from_matrix(left_hand_pose).as_rotvec().reshape(1, -1)
right_hand_pose = R.from_matrix(right_hand_pose).as_rotvec().reshape(1, -1)

body_pose       = torch.from_numpy(body_pose).float()
left_hand_pose  = torch.from_numpy(left_hand_pose).float()
right_hand_pose = torch.from_numpy(right_hand_pose).float()

output     = model(betas=betas,
                   body_pose=body_pose,
                   left_hand_pose=left_hand_pose,
                   right_hand_pose=right_hand_pose,
                   return_verts=True)

vertices   = output.vertices.detach().cpu().numpy().squeeze()
joints     = output.joints.detach().cpu().numpy().squeeze()

print(output)

vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.9]
tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)

# adding body meshs
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
scene = pyrender.Scene()
scene.add(mesh)

# adding body joints
sm = trimesh.creation.uv_sphere(radius=0.005)
sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
tfs = np.tile(np.eye(4), (len(joints), 1, 1))
tfs[:, :3, 3] = joints
joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
scene.add(joints_pcl)

pyrender.Viewer(scene, use_raymond_lighting=True)
