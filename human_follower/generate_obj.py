import bpy
import os

# bpy.ops.wm.addon_enable(module="io_scene_obj")  # ⬅ 启用 OBJ 导出支持
# === 用户配置 ===
name = "alien_soldier"
INPUT_MODEL = f"human_follower/human_model/{name}.glb"  # 支持 .glb / .fbx / .obj
OUTPUT_GLB = f"human_follower/human_model/{name}_convex.glb"

# === 清空当前场景 ===
bpy.ops.wm.read_factory_settings(use_empty=True)

# === 导入模型 ===
ext = os.path.splitext(INPUT_MODEL)[-1].lower()
if ext == ".glb" or ext == ".gltf":
    bpy.ops.import_scene.gltf(filepath=INPUT_MODEL)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=INPUT_MODEL)
elif ext == ".obj":
    bpy.ops.import_scene.obj(filepath=INPUT_MODEL)
else:
    raise ValueError("Unsupported format: " + ext)

# === 选中所有 mesh 并合并 ===
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
for obj in mesh_objects:
    obj.select_set(True)

bpy.context.view_layer.objects.active = mesh_objects[0]
bpy.ops.object.join()

# === 生成凸包 ===
joined_obj = bpy.context.view_layer.objects.active
bpy.ops.object.duplicate()
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.convex_hull()
bpy.ops.object.mode_set(mode='OBJECT')

# === 导出选中（凸包）为 .obj ===
convex_obj = bpy.context.view_layer.objects.active
for obj in bpy.context.scene.objects:
    obj.select_set(False)
convex_obj.select_set(True)

bpy.ops.export_scene.gltf(
    filepath=OUTPUT_GLB,
    export_format='GLB',
    use_selection=True,
    export_materials='NONE'
)
print(f"✅ Convex hull saved to {OUTPUT_GLB}")