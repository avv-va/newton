import os
from pxr import Usd, UsdGeom, Gf, Sdf
import trimesh
import argparse



def obj_to_usd(obj_path: str, usd_path: str, prim_path: str = "/root/mesh") -> None:
    """Convert a triangular OBJ file to a USD file containing a single mesh prim.

    Args:
        obj_path: Path to input OBJ file.
        usd_path: Path where the USD (.usd or .usda) file will be written.
        prim_path: The prim path inside the USD stage to place the mesh (default: /root/mesh).

    Raises:
        FileNotFoundError: if the OBJ file does not exist.
        ImportError: if the USD Python bindings (pxr) are not available.
        ValueError: if the input mesh is not triangular or cannot be loaded.
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")


    mesh = trimesh.load_mesh(obj_path, force='mesh')
    if mesh is None:
        raise ValueError(f"Failed to load mesh from: {obj_path}")

    if getattr(mesh, "faces", None) is None or mesh.faces.shape[1] != 3:
        raise ValueError(
            "Only triangular meshes are supported. If your OBJ contains polygonal faces, "
            "triangulate it beforehand (e.g. in Blender or with trimesh)."
        )

    verts = mesh.vertices
    faces = mesh.faces
    stage = Usd.Stage.CreateNew(usd_path)
    mesh_prim = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    points = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts]
    mesh_prim.CreatePointsAttr().Set(points)
    counts = [3] * len(faces)
    indices = [int(i) for tri in faces for i in tri]
    mesh_prim.CreateFaceVertexCountsAttr().Set(counts)
    mesh_prim.CreateFaceVertexIndicesAttr().Set(indices)
    mesh_prim.CreateDoubleSidedAttr().Set(True)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an OBJ file to a USD mesh.")
    parser.add_argument("--obj", help="Input OBJ file path")
    parser.add_argument("--usd", help="Output USD file path (.usd or .usda)")
    parser.add_argument("--prim", default="/root/mesh", help="Prim path to write the mesh to")
    args = parser.parse_args()

    obj_to_usd(args.obj, args.usd, prim_path=args.prim)