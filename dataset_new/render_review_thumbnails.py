"""Render one review thumbnail per GLB model with Blender.

Run with Blender, for example:
  blender --background --python dataset_new/render_review_thumbnails.py -- --dataset-dir dataset_new/Military_3D_Dataset
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "Military_3D_Dataset"


def parse_args() -> argparse.Namespace:
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render Objaverse review thumbnails.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_scene(resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items} else "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("ReviewWorld") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)


def import_model(model_path: Path) -> list[bpy.types.Object]:
    suffix = model_path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(model_path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(model_path))
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(model_path))
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    return mins, maxs


def normalize_model(objects: list[bpy.types.Object]) -> None:
    mins, maxs = scene_bounds(objects)
    center = (mins + maxs) * 0.5
    size = max((maxs - mins).x, (maxs - mins).y, (maxs - mins).z, 1e-6)
    scale = 2.0 / size
    for obj in objects:
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera_and_lights() -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 5.0))
    key = bpy.context.object
    key.name = "Review_Key_Light"
    key.data.energy = 500
    key.data.size = 5

    bpy.ops.object.light_add(type="POINT", location=(-3.0, 4.0, 3.0))
    fill = bpy.context.object
    fill.name = "Review_Fill_Light"
    fill.data.energy = 100

    bpy.ops.object.camera_add(location=(3.2, -4.0, 2.4))
    camera = bpy.context.object
    look_at(camera, Vector((0, 0, 0)))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 3.1
    bpy.context.scene.camera = camera
    return camera


def iter_models(dataset_dir: Path):
    for category_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        if category_dir.name.startswith("_"):
            continue
        for model_path in sorted(category_dir.glob("*.glb")):
            yield category_dir.name, model_path


def render_thumbnail(model_path: Path, out_path: Path, resolution: int) -> bool:
    clear_scene()
    setup_scene(resolution)
    objects = import_model(model_path)
    if not objects:
        return False
    normalize_model(objects)
    add_camera_and_lights()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)
    return True


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = (args.out_dir or dataset_dir / "_review_thumbnails").resolve()
    rendered = 0
    failed = 0

    for index, (category, model_path) in enumerate(iter_models(dataset_dir), start=1):
        if args.limit is not None and rendered >= args.limit:
            break
        out_path = out_dir / category / f"{model_path.stem}.png"
        if args.skip_existing and out_path.exists():
            continue
        print(f"[render] {index}: {model_path}")
        try:
            ok = render_thumbnail(model_path, out_path, args.resolution)
        except Exception as error:
            failed += 1
            print(f"[warn] Failed to render {model_path}: {error}")
            continue
        rendered += int(ok)

    print(f"[done] Rendered {rendered} thumbnails; failed {failed}; out={out_dir}")


if __name__ == "__main__":
    main()
