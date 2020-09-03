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

import pyrender
import trimesh
import numpy as np
import smplx
import torch


model = smplx.create(model_path="weights",
                     model_type="smplx",
                     gender="male",
                     use_face_contour=True,
                     num_betas=10,
                     num_expression_coeffs=10)

betas      = torch.randn([1, model.num_betas], dtype=torch.float32)
expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32)

output     = model(betas=betas, expression=expression, return_verts=True)
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
