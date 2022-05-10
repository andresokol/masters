import csv
import math
import os
import random
import struct
import sys
import typing as tp

sys.path.append("/home/andresokol/code/masters/venv/lib/python3.8/site-packages")

import bpy  # noqa: E402
import bmesh
import mathutils

import tqdm  # noqa: E402

PointT = tp.Tuple[float, float, float]

STRANDS_DIR = os.path.abspath('../mastersdata/models/hairsalon/')
RESULT_DIR = "./result2"


class Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_bytes(cls, file: tp.BinaryIO) -> "Vector":
        x, y, z = struct.unpack("<fff", file.read(12))
        return cls(x, -z, y)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> None:
        length = self.length()
        self.x /= length
        self.y /= length
        self.z /= length

    def __add__(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)
        return self.__class__(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)
        return self.__class__(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __rmul__(self, coeff: float) -> "Vector":
        return self.__class__(
            self.x * coeff,
            self.y * coeff,
            self.z * coeff,
        )

    def __mul__(self, other: float) -> "Vector":
        return self.__rmul__(other)

    def cross_product(self, other: "Vector") -> "Vector":
        assert isinstance(other, self.__class__)

        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x

        return Vector(x, y, z)

    def dot_product(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def tuple(self) -> tp.Tuple[float, float, float]:
        return self.x, self.y, self.z

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def get_any_normal(self) -> "Vector":
        SMALL = 1e-6
        if abs(self.x) < SMALL and abs(self.y) < SMALL:
            return self.__class__(0, -self.z, self.y)
        return self.__class__(-self.y, self.x, 0)


StrandT = tp.List[Vector]
EdgeT = tp.Tuple[int, int]
FaceT = tp.Tuple[int, ...]


def read_hair(filepath: str) -> tp.List[StrandT]:
    with open(filepath, "rb") as file:
        print(f"Reading file {filepath}")
        num_of_strands = struct.unpack("<i", file.read(4))[0]
        print(f"Number of strands: {num_of_strands}")

        total_vertices_count = 0
        strands: tp.List[StrandT] = []
        for _ in range(num_of_strands):
            strand_len = struct.unpack("<i", file.read(4))[0]
            total_vertices_count += strand_len

            strand: StrandT = [Vector.from_bytes(file) for _ in range(strand_len)]
            strands.append(strand)

        print(f"Total {total_vertices_count} vertices read")
        return strands


def build_normals(
        point_a: Vector,
        point_b: Vector,
        point_c: Vector,
        hair_width: float = 0.0003,
) -> tp.List[Vector]:
    vector_1 = point_a - point_b
    vector_2 = point_c - point_b

    vector_1.normalize()
    vector_2.normalize()

    if vector_1.dot_product(vector_2) > 0.0001:
        median = vector_1 + vector_2
    else:
        median = vector_1.get_any_normal()
    median.normalize()
    opposed_median = -1 * median

    normal = vector_1.cross_product(median)
    normal.normalize()

    opposed_normal = -1 * normal

    points = [
        point_b + hair_width * median,
        point_b + hair_width * normal,
        point_b + hair_width * opposed_median,
        point_b + hair_width * opposed_normal,
    ]

    # Rotate points to prevent skewing
    max_x = points[0].x
    max_x_idx = 0
    for i in range(1, 4):
        if points[i].x > max_x:
            max_x = points[i].x
            max_x_idx = i

    return points[max_x_idx:] + points[:max_x_idx]


def build_mesh_from_strand(
        strand: StrandT,
) -> tp.Tuple[tp.List[PointT], tp.List[EdgeT], tp.List[FaceT], tp.List[PointT]]:
    if len(strand) <= 2:
        return [], [], [], []

    vertices: tp.List[PointT] = [strand[0].tuple()]
    colors: tp.List[PointT] = [(0, 0, 0)]

    for i in range(1, len(strand) - 1):
        vertices += [
            x.tuple() for x in build_normals(strand[i - 1], strand[i], strand[i + 1])
        ]

        direction = strand[i] - strand[i - 1]
        direction.normalize()
        colors += [(abs(direction.x), abs(direction.y), abs(direction.z))] * 4

    vertices += [strand[-1].tuple()]
    colors += [(0, 0, 0)]

    edges: tp.List[EdgeT] = [(0, 1), (0, 2), (0, 3), (0, 4)]
    faces: tp.List[FaceT] = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]

    for i in range(len(strand) - 3):
        idx = 4 * i + 1
        # segment circular
        edges += [
            (idx, idx + 1),
            (idx + 1, idx + 2),
            (idx + 2, idx + 3),
            (idx + 3, idx),
        ]

        # cross-segment direct
        edges += [
            (idx, idx + 4),
            (idx + 1, idx + 5),
            (idx + 2, idx + 6),
            (idx + 3, idx + 7),
        ]

        # cross-segment diagonal
        edges += [
            (idx, idx + 5),
            (idx + 1, idx + 6),
            (idx + 2, idx + 7),
            (idx + 3, idx + 4),
        ]

        faces += [
            (idx, idx + 4, idx + 5),
            (idx, idx + 5, idx + 1),
            (idx + 1, idx + 5, idx + 6),
            (idx + 1, idx + 6, idx + 2),
            (idx + 2, idx + 6, idx + 7),
            (idx + 2, idx + 7, idx + 3),
            (idx + 3, idx + 7, idx + 4),
            (idx + 3, idx + 4, idx),
        ]

    last_idx = 1 + 4 * (len(strand) - 3)

    edges += [
        # last segment
        (last_idx, last_idx + 1),
        (last_idx + 1, last_idx + 2),
        (last_idx + 2, last_idx + 3),
        (last_idx + 3, last_idx),
        # conical
        (last_idx, last_idx + 4),
        (last_idx + 1, last_idx + 4),
        (last_idx + 2, last_idx + 4),
        (last_idx + 3, last_idx + 4),
    ]

    faces += [
        (last_idx, last_idx + 1, last_idx + 4),
        (last_idx + 1, last_idx + 2, last_idx + 4),
        (last_idx + 2, last_idx + 3, last_idx + 4),
        (last_idx + 3, last_idx, last_idx + 4),
    ]

    assert len(vertices) == (
            len(strand) - 2) * 4 + 2, f'Incorrect number of vertices: {len(vertices)}, {(len(strand) - 2) * 4 + 2}'

    return vertices, edges, faces, colors


# def draw_debug_points():
#     me = bpy.data.meshes.new("Landmarks mesh")
#     ob = bpy.data.objects.new("Landmarks", me)
#
#     coords = [
#         (pt[0], -pt[2], pt[1])
#         for pt in DEBUG_POINTS
#     ]
#
#     me.from_pydata(coords, [], [])
#     me.update()
#     bpy.context.collection.objects.link(ob)


def load_to_object(strands: tp.List[StrandT]):
    all_coords: tp.List[PointT] = []
    all_edges: tp.List[EdgeT] = []
    all_faces: tp.List[FaceT] = []
    all_colors: tp.List[PointT] = []
    idx_offset = 0

    invalid_strands_cnt = 0

    print("Constructing mesh from strands")
    for strand in tqdm.tqdm(strands):
        coords, edges, faces, colors = build_mesh_from_strand(strand)

        if not coords:
            invalid_strands_cnt += 1
            continue

        for st_idx, end_idx in edges:
            all_edges.append((st_idx + idx_offset, end_idx + idx_offset))
        for face in faces:
            all_faces.append(tuple(idx + idx_offset for idx in face))
        all_coords += coords
        all_colors += colors
        idx_offset += len(coords)

    print(f"Number of invalid strands: {invalid_strands_cnt}")

    print("Loading mesh to blender object...")
    me = bpy.data.meshes.new("Hair mesh")
    ob = bpy.data.objects.new("Hair", me)
    me.from_pydata(all_coords, all_edges, all_faces)
    print("Mesh had errors: ", me.validate(verbose=True))

    # Ignored by Cycles engine
    # for p in me.polygons:
    #     p.use_smooth = True
    me.update()

    # coloring vertices
    bm = bmesh.new()
    bm.from_mesh(me)
    color_layer = bm.loops.layers.color.new("color")
    for face in bm.faces:
        for loop in face.loops:
            loop[color_layer] = (*all_colors[loop.vert.index], 1.0)
    bm.to_mesh(me)
    me.update()

    print("Loaded:")
    print(f"  {len(ob.data.vertices)} vertices")
    print(f"  {len(ob.data.edges)} edges")
    print(f"  {len(ob.data.polygons)} polygons")
    return ob


def create_physical_material():
    mat = bpy.data.materials.new(name="Hair Physical Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    geo_input_node = nodes.new(type="ShaderNodeNewGeometry")

    hair_node = nodes.new(type="ShaderNodeBsdfHairPrincipled")
    hair_node.parametrization = "MELANIN"
    hair_node.inputs["Melanin"].default_value = 0.5
    hair_node.inputs["Melanin Redness"].default_value = 0.5
    hair_node.inputs["Roughness"].default_value = 0.5
    hair_node.inputs["Radial Roughness"].default_value = 0.3
    hair_node.inputs["Coat"].default_value = 0.3
    hair_node.inputs["Random Color"].default_value = 0.5
    hair_node.inputs["Random Roughness"].default_value = 0.5

    output_node_cycles = nodes.new(type="ShaderNodeOutputMaterial")
    output_node_cycles.target = "CYCLES"

    links.new(geo_input_node.outputs["Random Per Island"], hair_node.inputs["Random"])
    links.new(hair_node.outputs["BSDF"], output_node_cycles.inputs["Surface"])

    # # Eevee path
    # vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    #
    # output_node_eevee = nodes.new(type="ShaderNodeOutputMaterial")
    # output_node_eevee.target = "EEVEE"
    #
    # links.new(vertex_color_node.outputs["Color"], output_node_eevee.inputs["Surface"])

    return mat


def create_structure_material():
    mat = bpy.data.materials.new(name="Hair Structure Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    input_node = nodes.new(type="ShaderNodeVertexColor")

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.target = "CYCLES"

    links.new(input_node.outputs["Color"], output_node.inputs["Surface"])

    return mat


#
# def add_subdivision_surface(obj):
#     modifier = obj.modifiers.new(name="Subdivision Surface", type="SUBSURF")
#     bpy.ops.object.modifier_apply(modifier=modifier.name)


def setup_render_engine():
    scene = bpy.context.scene
    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 32
    scene.cycles.samples = 64
    scene.cycles.max_bounces = 4
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.image_settings.file_format = "PNG"

    # data required by compositor
    # bpy.context.view_layer.use_pass_object_index = True
    # bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_cryptomatte_object = True


def setup_compositor():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    # clean-up
    for node in tree.nodes:
        tree.nodes.remove(node)

    input_node = tree.nodes.new(type="CompositorNodeRLayers")

    # id_mask_node = tree.nodes.new(type="CompositorNodeIDMask")
    # id_mask_node.index = 2
    # id_mask_node.use_antialiasing = True
    #
    # set_alpha_node = tree.nodes.new(type="CompositorNodeSetAlpha")

    cryptomatte_node = tree.nodes.new(type="CompositorNodeCryptomatteV2")
    cryptomatte_node.matte_id = "Hair"

    # base_output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    # base_output_node.base_path = "./output_base"
    base_output_node = tree.nodes.new(type="CompositorNodeComposite")
    base_output_node.use_alpha = True

    tree.links.new(input_node.outputs["Image"], cryptomatte_node.inputs["Image"])
    tree.links.new(cryptomatte_node.outputs["Image"], base_output_node.inputs["Image"])

    # tree.links.new(input_node.outputs["IndexOB"], id_mask_node.inputs["ID value"])
    # tree.links.new(input_node.outputs["Image"], set_alpha_node.inputs["Image"])
    # tree.links.new(id_mask_node.outputs["Alpha"], set_alpha_node.inputs["Alpha"])
    # tree.links.new(set_alpha_node.outputs["Image"], base_output_node.inputs["Image"])

    # sep_rgba_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    # tree.links.new(input_node.outputs["Normal"], sep_rgba_node.inputs["Image"])
    #
    # # R chanel for vertical, 0 - down, 1 up
    # add_r_node = tree.nodes.new(type="CompositorNodeMath")
    # add_r_node.operation = "ADD"
    # add_r_node.inputs[1].default_value = 1
    # tree.links.new(sep_rgba_node.outputs["R"], add_r_node.inputs[0])
    #
    # mul_r_node = tree.nodes.new(type="CompositorNodeMath")
    # mul_r_node.operation = "MULTIPLY"
    # mul_r_node.inputs[1].default_value = 0.5
    # tree.links.new(add_r_node.outputs["Value"], mul_r_node.inputs[0])
    #
    # # B channel for horizontal, 0 - left, 1 - right
    # add_b_node = tree.nodes.new(type="CompositorNodeMath")
    # add_b_node.operation = "ADD"
    # add_b_node.inputs[1].default_value = 1
    # tree.links.new(sep_rgba_node.outputs["B"], add_b_node.inputs[0])
    #
    # mul_b_node = tree.nodes.new(type="CompositorNodeMath")
    # mul_b_node.operation = "MULTIPLY"
    # mul_b_node.inputs[1].default_value = 0.5
    # tree.links.new(add_b_node.outputs["Value"], mul_b_node.inputs[0])
    #
    # combine_rgba_node = tree.nodes.new(type="CompositorNodeCombRGBA")
    # tree.links.new(mul_r_node.outputs["Value"], combine_rgba_node.inputs["R"])
    # tree.links.new(mul_b_node.outputs["Value"], combine_rgba_node.inputs["B"])
    #
    # set_alpha_node_2 = tree.nodes.new(type="CompositorNodeSetAlpha")
    # tree.links.new(combine_rgba_node.outputs["Image"], set_alpha_node_2.inputs["Image"])
    # tree.links.new(id_mask_node.outputs["Alpha"], set_alpha_node_2.inputs["Alpha"])
    #
    # normals_output_node = tree.nodes.new("CompositorNodeOutputFile")
    # normals_output_node.base_path = "./output_normals"
    # tree.links.new(set_alpha_node_2.outputs["Image"], normals_output_node.inputs["Image"])


def render(filename: str):
    # print("Rendering...")
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.filepath = f"{RESULT_DIR}/{filename}.png"
    bpy.ops.render.render(write_still=1)

    # bpy.context.scene.render.engine = "BLENDER_EEVEE"
    # bpy.context.scene.render.filepath = f"{RESULT_DIR}/{filename}_structure.png"
    # bpy.ops.render.render(write_still=1)
    # print("Render done!")


# def draw(filepath: str):
#     mat = create_material()
#     setup_render_engine()
#     setup_compositor()
#
#     strands = read_hair(filepath)
#
#     obj = load_to_object(strands[::10])
#     obj.pass_index = 2  # for compositor
#     bpy.context.collection.objects.link(obj)
#     bpy.context.view_layer.objects.active = obj
#
#     obj.location = (0, 1, -1.745)
#     obj.data.materials.append(mat)
#
#     render("test")


def batch(iterable: tp.Sized, step: int = 1):
    length = len(iterable)
    for ndx in range(0, length, step):
        yield iterable[ndx:min(ndx + step, length)]


def main():
    strand_names = [name[:-5] for name in os.listdir(STRANDS_DIR) if name.endswith('.data')]
    strand_names.sort()
    print(f"Found {len(strand_names)} strands files")

    physical_mat = create_physical_material()
    structure_mat = create_structure_material()

    setup_render_engine()
    setup_compositor()

    with open("head_positions.csv") as file:
        reader = csv.reader(file)
        head_positions = [row for row in reader]
    head_positions = head_positions[1:]  # remove header
    print(f"Read {len(head_positions)} head positions")

    for i, heads_batch in tqdm.tqdm(enumerate(batch(head_positions, step=1))):
        strands = read_hair(os.path.join(STRANDS_DIR, f'{strand_names[i % len(strand_names)]}.data'))

        hair_obj = load_to_object(strands[::10])
        bpy.context.collection.objects.link(hair_obj)
        bpy.context.view_layer.objects.active = hair_obj

        hair_obj.location = (0.00471, 1.08974, -1.745958)
        hair_obj.data.materials.append(physical_mat)

        bpy.data.objects['Hair'].select_set(True)
        bpy.context.scene.cursor.location = mathutils.Vector((0.0, 1.0, 0.0))
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

        for mat, name_suffix in ((physical_mat, "base"), (structure_mat, "structure")):
            hair_obj.data.materials[0] = mat
            print(f"Rendering {name_suffix} batch")

            for position in tqdm.tqdm(heads_batch):
                imgdir, imgname, *params = position
                x, y, z, rot_x, rot_y, rot_z = list(map(float, params))

                location = mathutils.Vector((x, y, z))
                rotation = mathutils.Vector((rot_x, rot_y, rot_z))

                for object_id in ('head_model', 'Torso', 'Hair'):
                    bpy.data.objects[object_id].location = location
                    bpy.data.objects[object_id].rotation_euler = rotation

                render(f"{imgdir}_{imgname}_{name_suffix}")

        bpy.data.objects.remove(hair_obj)
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)


main()
