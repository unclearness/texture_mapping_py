import open3d as o3d
import numpy as np
import os
from visibility_tester import loadKeyframes, VisibilityTester
import time


def saveObj(
    filePath, vertices, faceVertIDs, uvs=[], normals=[],
        uvIDs=[], normalIDs=[], vertexColors=[], mat_file=None, mat_name=None):
    f_out = open(filePath, "w")
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# Vertices: %s\n" % (len(vertices)))
    f_out.write("# Faces: %s\n" % (len(faceVertIDs)))
    f_out.write("#\n")
    f_out.write("####\n")
    if mat_file is not None:
        f_out.write("mtllib " + mat_file + "\n")
    for vi, v in enumerate(vertices):
        vStr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s" % (color[0], color[1], color[2])
        vStr += "\n"
        f_out.write(vStr)
    f_out.write("# %s vertices\n\n" % (len(vertices)))
    for uv in uvs:
        uvStr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvStr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    if mat_name is not None:
        f_out.write("usemtl " + mat_name + "\n")
    for fi, fvID in enumerate(faceVertIDs):
        fStr = "f"
        for fvi, fvIDi in enumerate(fvID):
            fStr += " %s" % (fvIDi + 1)
            if len(uvIDs) > 0:
                fStr += "/%s" % (uvIDs[fi][fvi] + 1)
            if len(normalIDs) > 0:
                fStr += "/%s" % (normalIDs[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(faceVertIDs)))
    f_out.write("# End of File\n")
    f_out.close()


if __name__ == "__main__":
    obj_path = "./data/bunny/GT/bunny.obj"
    tum_path = "./data/bunny/input/tumpose.txt"
    intrin_path = "./data/bunny/input/intrin.txt"
    img_dir = "./data/bunny/input/"
    img_names = sorted([x for x in os.listdir(img_dir) if x.endswith(".png")])
    img_paths = [os.path.join(img_dir, x) for x in img_names]
    kfs = loadKeyframes(tum_path, intrin_path, img_paths)
    tester = VisibilityTester()
    tester.init(obj_path)
    start = time.time()
    for kf in kfs:
        tester.test(kf)
    end = time.time()
    print(f"Visibility test for {len(kfs)} frames", end - start)
    start = time.time()
    vertex_colors = tester.computeVertexColor()
    end = time.time()
    print(f"Vertex color computation", end - start)
    saveObj("./data/bunny/out.obj", tester.vertices,
            tester.triangles, vertexColors=vertex_colors)
