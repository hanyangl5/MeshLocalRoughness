# -*- coding: utf-8 -*-
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from vispy.scene import visuals
import vispy.scene
from numpy import *
import numpy as np
from scipy.sparse import dok_matrix
import math

vertices = []
faces = []
vn = []
fn = []

# read .off file


def read_off(path):
    with open(path, 'r') as file:

      text = file.read()

      text = text.lstrip()
      header, raw = text.split(None, 1)
      if header.upper() not in ['OFF', 'COFF']:
          print("invalid file type")
          exit(-1)
      # remove whitespace,comment
      splits = [i.strip() for i in str.splitlines(str(raw))]
      splits = [i for i in splits if len(i) > 0 and i[0] != '#']
      # get vertexnum facenum
      header = np.array(splits[0].split(), dtype=np.int64)
      vertex_count, face_count = header[:2]
      # get vertices
      vertices = np.array([
          i.split()[:3] for i in
          splits[1: vertex_count + 1]],
          dtype=np.float64)
      vertices = vertices.reshape((vertex_count, 3))
      # get faces
      faces = [i.split()
               for i in splits[vertex_count + 1:vertex_count + face_count + 1]]
      faces = [line[1:int(line[0]) + 1] for line in faces]
      faces = np.array(faces, dtype=np.int64)
      return vertices, faces, vertex_count, face_count


def callr():
  angle = np.zeros((3, fn))
  ctan = np.zeros((3, fn))
  sum_angles = np.zeros((vn))
  constants = np.tile(2*math.pi, (vn))
  D = dok_matrix((vn, vn))
  epsilon = np.tile(1e-10, (vn))
  a = 0.15

  for i in range(3):

    # generate 3 different squence of indices
    i1 = (i) % 3
    i2 = (i+1) % 3
    i3 = (i+2) % 3

    # adjacency edge vector
    pp = vertices[faces[:, i2]] - vertices[faces[:, i1]]
    qq = vertices[faces[:, i3]] - vertices[faces[:, i1]]

    # normorlization
    pp_length = np.sqrt(sum(np.multiply(pp, pp), 1))
    qq_length = np.sqrt(sum(np.multiply(qq, qq), 1))

    for j in range(len(pp_length)):
      if pp_length[j] < 1e-10:
        pp_length[j] = 1
      if qq_length[j] < 1e-10:
        qq_length[j] = 1
      else:
        pass

    pp_nor = pp/np.tile(pp_length, (3, 1)).transpose()
    qq_nor = qq/np.tile(qq_length, (3, 1)).transpose()

    # calculate angles
    cos_ang = sum(np.multiply(pp_nor, qq_nor), 1)
    np.clip(cos_ang, -1, 1, cos_ang)
    angle[i] = np.arccos(cos_ang)
    tan = np.tan(angle[i])

    # prevent to divide zero
    for k in range(len(tan)):
      if abs(tan[k]-0) < 1e-10:
        tan[k] = 1e-16

    ctan[i] = (1/tan)/2

    np.clip(ctan[i], 0.0001, 1000, ctan[i])

    for j in range(fn):
        temp = faces[j, i]
        sum_angles[temp] += angle[i, j]

    # adjacency matrix
    D[faces[:, i2], faces[:, i3]] = ctan[i]

  # Gaussisan Curavture of each vertex
  GC = abs(constants-sum_angles)

  # Local Roughness of each vertex
  LR = np.array(abs(GC-(D*GC)/sum(D, axis=0))).flatten()
  # fix value
  np.clip(LR, 0.0005, 0.2, LR)
  LR[:] = pow(LR[:], a)

  # Output
  return LR


def map_roughness_to_color(roughness_values, min, denom):
    cmap = matplotlib.colormaps.get_cmap('viridis')
    normalized_values = (roughness_values - min) / \
        (denom)
    colors = cmap(normalized_values)
    return colors


def visualize(roughness_values):
  min = np.min(roughness_values)
  denom = (np.max(roughness_values) - np.min(roughness_values))

  roughness = [map_roughness_to_color(
      float(value), min, denom) for value in roughness_values]
  roughness_values = np.array(roughness)

  canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
  view = canvas.central_widget.add_view()

  mesh = visuals.Mesh(vertices=vertices, faces=faces,
                      vertex_colors=roughness_values, shading='smooth')

  view.add(mesh)

  view.camera = 'turntable'

  vispy.app.run()


if __name__ == '__main__':
  path = 'test/venus.off'

  # load mesh
  (vertices, faces, vn, fn) = read_off(path)
  print('mesh loaded', 'vertices', vn, 'faces', fn)

  # compute roughness
  lr = callr()
  print('roughness', lr)

  # visualize
  visualize(lr)
