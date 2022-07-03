import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as Rot
from dataclasses import dataclass


@dataclass
class Keyframe:
    id: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w_R: np.array
    c2w_t: np.array
    img: np.array


def loadTum(path):
    timestamps = []
    Rs = []
    ts = []
    additionals = {}
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            splited = line.rstrip().split(' ')
            timestamp = int(splited[0])
            tq = [float(x) for x in splited[1:8]]
            t = np.array(tq[0:3])
            q = Rot.from_quat(tq[3:])
            timestamps.append(timestamp)
            Rs.append(q.as_matrix())
            ts.append(t)
            if 8 < len(splited):
                additionals.append(splited[8:])
    return timestamps, Rs, ts, additionals


def loadIntrin(path):
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            splited = line.rstrip().split(' ')
            width = int(splited[0])
            height = int(splited[1])
            fx = float(splited[2])
            fy = float(splited[3])
            cx = float(splited[4])
            cy = float(splited[5])
    return width, height, fx, fy, cx, cy


def loadKeyframes(tum_path, intrin_path, img_paths):
    timestamps, Rs, ts, additionals = loadTum(tum_path)
    width, height, fx, fy, cx, cy = loadIntrin(intrin_path)
    kfs = []
    for i, (R, t, img_path) in enumerate(zip(Rs, ts, img_paths)):
        img = cv2.imread(img_path)[..., ::-1]
        kf = Keyframe(i, fx, fy, cx, cy, R, t, img)
        kfs.append(kf)
    return kfs


def project(points, fx, fy, cx, cy, w2c_R, w2c_t):
    # https://stackoverflow.com/questions/68422297/batch-matrix-multiplication-in-numpy
    points_c = np.matmul(w2c_R, points[:, :, None]).squeeze(-1) + w2c_t
    # print(points_c[0], w2c_R @ points[0] + w2c_t)
    x = points_c[..., 0]
    y = points_c[..., 1]
    z = points_c[..., 2]
    u = x * fx / z + cx
    v = y * fy / z + cy
    return np.stack([u, v], axis=-1)


def unproject(u, v, d, fx, fy, cx, cy):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return np.stack([x, y, d], axis=-1)


def rayFromPixel(u, v, fx, fy, cx, cy):
    d = np.ones_like(u)
    dir = unproject(u, v, d, fx, fy, cx, cy)
    ray = dir / np.linalg.norm(dir, axis=-1, keepdims=True)
    return ray


@dataclass
class VertexInfoPerKeyframe:
    id: int
    ray: np.array
    color: np.array
    intensity: float
    proj_pos: np.array
    viewing_angle: float
    distance: float


@dataclass
class VertexInfo:
    visible_keyframes: list


def color2gray(r, g, b):
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class VisibilityTester:
    def __init__(self):
        pass

    def init(self, obj_path):
        self.obj_path = obj_path
        self.mesh = o3d.io.read_triangle_mesh(obj_path)
        self.mesh.compute_vertex_normals()
        self.mesh_for_raycast = o3d.t.geometry.TriangleMesh.from_legacy(
            self.mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.mesh_id = self.scene.add_triangles(self.mesh_for_raycast)
        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)
        self.bb = self.mesh.get_axis_aligned_bounding_box()
        self.eps = np.min(self.bb.get_extent()) * 1e-3

        self.vertex_info = [VertexInfo([]) for _ in range(len(self.vertices))]

    def updateVertexInfo(self, id, img, visibile_vertices, distances, rays_w,
                         vertex_normals, projected, biliner=True):
        prj_pos_list = projected[visibile_vertices]
        prj_pos_list_int = prj_pos_list.astype(int)
        # print(prj_pos_list_int, prj_pos_list_int.shape,
        #      np.max(prj_pos_list_int[..., 0]), np.max(prj_pos_list_int[..., 1]), img.shape)
        # colors = np.take_along_axis(img, prj_pos_list_int, axis=-1)
        colors = img[prj_pos_list_int[..., 1], prj_pos_list_int[..., 0]]
        dot_product = (
            rays_w[visibile_vertices] *
            (-1 * vertex_normals[visibile_vertices])).sum(axis=-1)
        viewing_angles = np.arccos(dot_product)
        intensities = color2gray(
            colors[..., 2], colors[..., 1], colors[..., 0])
        visibile_indices = np.where(visibile_vertices)[0]
        # print(visibile_indices)
        for index, ray, proj_pos, color, intensity, viewing_angle, distance in\
                zip(visibile_indices, rays_w,
                    prj_pos_list, colors, intensities,
                    viewing_angles, distances[visibile_vertices]):
            vi_kf = VertexInfoPerKeyframe(
                id, ray, color, intensity, proj_pos, viewing_angle, distance)
            self.vertex_info[index].visible_keyframes.append(vi_kf)

    def test(self, kf: Keyframe):
        h, w, _ = kf.img.shape
        if False:
            h, w, _ = kf.img.shape
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                width_px=w,
                height_px=h,
            )
        V = len(self.vertices)
        # Project all vertices
        w2c_R = kf.c2w_R.T
        w2c_t = - w2c_R @ kf.c2w_t
        projected = project(self.vertices, kf.fx, kf.fy,
                            kf.cx, kf.cy, w2c_R, w2c_t)
        inside_mask = (0 <= projected[..., 0]) & (
            projected[..., 0] <= w - 1) & (0 <= projected[..., 1]) &\
            (projected[..., 1] <= h - 1)

        # Test occlusions for projected points
        rays_c = rayFromPixel(projected[..., 0], projected[..., 1],
                              kf.fx, kf.fy,
                              kf.cx, kf.cy)
        rays_w = np.matmul(kf.c2w_R, rays_c[:, :, None]).squeeze(-1)
        # print(kf.c2w_R @ rays_c[0], rays_w[0], rays_w.shape)
        # origin = self.vertices  # np.repeat(kf.c2w_t[None], V, axis=0)
        origin = np.repeat(kf.c2w_t[None], V, axis=0)

        rays_w_origin = np.zeros((V, 6), dtype=np.float32)
        rays_w_origin[..., 0:3] = origin.astype(np.float32)
        rays_w_origin[..., 3:6] = rays_w.astype(np.float32)
        ans = self.scene.cast_rays(rays_w_origin)
        pos = ans['t_hit'].numpy()[..., None] * rays_w + origin
        visibile_vertices = (
            np.abs(pos - self.vertices).sum(axis=-1) < self.eps)

        visibile_vertices = visibile_vertices & inside_mask

        self.updateVertexInfo(
            kf.id, kf.img, visibile_vertices, ans['t_hit'], rays_w,
            self.vertex_normals,
            projected)

    def computeVertexColor(self, ignore_color=[0, 0, 0],
                           aggr_type='min_angle', calc_func=None):
        def mode(xs, viewing_angle):
            uniqs, counts = np.unique(xs, return_counts=True, axis=0)
            return uniqs[np.where(counts == np.amax(counts))[0][0]]

        def average(xs, viewing_angles=None):
            if viewing_angles is None:
                return np.average(xs, axis=0)
            inv_viewing_angles = 1.0 / viewing_angles
            denom = inv_viewing_angles.sum()
            weights = (inv_viewing_angles / denom)[..., None]
            return np.average(xs * weights, axis=0)

        def minAngle(xs, viewing_angle):
            index = np.argmin(viewing_angle)
            return xs[index]

        if aggr_type == 'mode':
            aggr_func = mode
        elif aggr_type == 'average':
            aggr_func = average
        elif aggr_type == 'min_angle':
            aggr_func = minAngle

        vertex_colors = []
        for vi in self.vertex_info:
            if calc_func is not None:
                vc = calc_func(vi)
            else:
                if len(vi.visible_keyframes) < 1:
                    vc = [0, 0, 0]
                else:
                    colors = np.array([x.color for x in vi.visible_keyframes])
                    viewing_angles = np.array(
                        [x.viewing_angle for x in vi.visible_keyframes])
                    if ignore_color is not None:
                        ignore_indices = (
                            np.sum((colors == ignore_color), axis=-1) == 3)
                        ok_indices = np.bitwise_not(ignore_indices)
                        colors = colors[ok_indices]
                        viewing_angles = viewing_angles[ok_indices]
                    if len(colors) < 1:
                        vc = [0, 0, 0]
                    else:
                        vc = aggr_func(colors, viewing_angles)
            vertex_colors.append(vc)
        return vertex_colors
