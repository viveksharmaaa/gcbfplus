import bpy
import math
import os
from mathutils import Vector, Euler

# ============================================================
# USER SETTINGS
# ============================================================

FPS = 30
DURATION = 20.0
N_FRAMES = int(FPS * DURATION)

OUTPUT_FILE = "//f16_formation_corridor.mp4"
MODEL_PATH = "/assets/f16_.glb"

# Set this to a .blend/.obj/.fbx aircraft model if desired.
# Leave as None to use the procedural aircraft model below.
AIRCRAFT_MODEL = None

CORRIDOR_LENGTH = 100.0
CORRIDOR_WIDTH = 18.0
CORRIDOR_HEIGHT = 12.0

FORMATION_SCALE = 3.7

# ============================================================
# RESET SCENE
# ============================================================

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene

scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

scene.render.filepath = OUTPUT_FILE
scene.render.fps = FPS

scene.frame_start = 1
scene.frame_end = N_FRAMES

scene.world.color = (0.03, 0.03, 0.05)

# ============================================================
# MATERIAL
# ============================================================

def material(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)

    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

    return mat


MAT_FLOOR = material(
    "Floor",
    (0.04, 0.25, 0.035),
    roughness=0.8
)

MAT_WALL = material(
    "Wall",
    (0.45, 0.38, 0.18),
    roughness=0.7
)

MAT_OBSTACLE = material(
    "Obstacle",
    (0.32, 0.28, 0.18),
    roughness=0.65
)

MAT_JET = material(
    "Jet",
    (0.12, 0.14, 0.16),
    metallic=0.65,
    roughness=0.3
)

MAT_CANOPY = material(
    "Canopy",
    (0.03, 0.08, 0.10),
    metallic=0.2,
    roughness=0.15
)

# ============================================================
# HELPERS
# ============================================================

def create_cube(name, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(location=location)

    obj = bpy.context.object
    obj.name = name
    obj.scale = scale

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True
    )

    obj.data.materials.append(mat)

    return obj


# ============================================================
# FLOOR
# ============================================================

floor = create_cube(
    "Ground",
    (0, CORRIDOR_LENGTH / 2, -0.3),
    (
        CORRIDOR_WIDTH / 2,
        CORRIDOR_LENGTH / 2,
        0.3
    ),
    MAT_FLOOR
)

# ============================================================
# WALLS
# ============================================================

left_wall = create_cube(
    "LeftWall",
    (
        -CORRIDOR_WIDTH / 2 - 0.25,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    (
        0.25,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    MAT_WALL
)

right_wall = create_cube(
    "RightWall",
    (
        CORRIDOR_WIDTH / 2 + 0.25,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    (
        0.25,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    MAT_WALL
)

# ============================================================
# CEILING BOUNDARY
# Semi-transparent-looking beam sections could be added,
# but we keep the top visually open.
# ============================================================

# ============================================================
# OBSTACLES
# ============================================================

obstacles = [

    # x, y, z, sx, sy, sz

    (-4.5, 20, 2.0, 2.0, 2.0, 2.0),

    (4.0, 32, 3.5, 2.0, 2.0, 3.5),

    (0.0, 43, 2.5, 2.4, 2.0, 2.5),

    (-4.8, 56, 4.8, 2.0, 2.0, 4.8),

    (4.5, 68, 2.5, 2.0, 2.0, 2.5),

    (0.0, 80, 5.0, 2.5, 2.0, 5.0),
]

for i, obs in enumerate(obstacles):

    x, y, z, sx, sy, sz = obs

    create_cube(
        f"Obstacle_{i}",
        (x, y, z),
        (sx, sy, sz),
        MAT_OBSTACLE
    )

# ============================================================
# PROCEDURAL FIGHTER AIRCRAFT
# ============================================================

def import_f16(name):

    # GLB / GLTF
    bpy.ops.import_scene.gltf(filepath=MODEL_PATH)

    imported = list(bpy.context.selected_objects)

    # Parent every imported component to one root object
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)

    for obj in imported:
        obj.parent = root

    # Adjust once after inspecting the imported model
    root.scale = (0.8, 0.8, 0.8)

    # These may need adjustment depending on how the model is oriented.
    root.rotation_euler = (
        0.0,
        0.0,
        0.0,
    )

    return root

def create_fighter_(name):

    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)

    # --------------------------
    # Fuselage
    # --------------------------

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=0.32,
        depth=3.5,
        location=(0, 0, 0)
    )

    fuselage = bpy.context.object

    # cylinder points along Z by default;
    # rotate so aircraft points along +Y
    fuselage.rotation_euler.x = math.radians(90)

    fuselage.parent = root
    fuselage.data.materials.append(MAT_JET)

    # --------------------------
    # Nose
    # --------------------------

    bpy.ops.mesh.primitive_cone_add(
        vertices=32,
        radius1=0.32,
        radius2=0.02,
        depth=1.3,
        location=(0, 2.35, 0)
    )

    nose = bpy.context.object
    nose.rotation_euler.x = math.radians(-90)
    nose.parent = root
    nose.data.materials.append(MAT_JET)

    # --------------------------
    # Main wings
    # --------------------------

    wing = create_cube(
        name + "_Wing",
        (0, 0.05, 0),
        (2.3, 0.75, 0.07),
        MAT_JET
    )

    wing.parent = root

    # --------------------------
    # Tail horizontal stabilizer
    # --------------------------

    tail = create_cube(
        name + "_Tail",
        (0, -1.35, 0.08),
        (1.15, 0.35, 0.06),
        MAT_JET
    )

    tail.parent = root

    # --------------------------
    # Vertical stabilizer
    # --------------------------

    fin = create_cube(
        name + "_Fin",
        (0, -1.15, 0.55),
        (0.08, 0.45, 0.65),
        MAT_JET
    )

    fin.rotation_euler.x = math.radians(-15)
    fin.parent = root

    # --------------------------
    # Canopy
    # --------------------------

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        location=(0, 0.7, 0.35)
    )

    canopy = bpy.context.object

    canopy.scale = (
        0.28,
        0.65,
        0.22
    )

    canopy.parent = root
    canopy.data.materials.append(MAT_CANOPY)

    # --------------------------
    # Scale whole aircraft
    # --------------------------

    root.scale = (0.75, 0.75, 0.75)

    return root


# ============================================================
# CREATE FOUR AIRCRAFT
# ============================================================

jets = []

for i in range(4):

    jet = create_fighter(f"F16_{i+1}")
    jets.append(jet)

# ============================================================
# DIAMOND FORMATION
#
# coordinate convention:
#
# x = lateral
# y = forward
# z = altitude
#
# ============================================================

formation_offsets = [

    Vector((0.0, 3.2, 0.0)),       # leader

    Vector((-FORMATION_SCALE, 0, 0)),  # left

    Vector((FORMATION_SCALE, 0, 0)),   # right

    Vector((0.0, -3.6, 0.0)),      # rear
]

# ============================================================
# SMOOTH TRAJECTORY UTILITIES
# ============================================================

def smoothstep(x):

    x = max(0.0, min(1.0, x))

    return (
        3 * x**2
        - 2 * x**3
    )


def blend(t, t0, t1):

    if t <= t0:
        return 0.0

    if t >= t1:
        return 1.0

    return smoothstep(
        (t - t0) / (t1 - t0)
    )


# ============================================================
# FORMATION CENTER TRAJECTORY
#
# This defines a series of diverse maneuvers.
# ============================================================

def center_trajectory(t):

    # constant forward motion
    y = 5.0 + 4.4 * t

    x = 0.0
    z = 4.0

    # ------------------------------------------------
    # Maneuver 1:
    # lateral left dodge
    # ------------------------------------------------

    x += -3.5 * blend(t, 2.0, 4.0)

    # recover
    x += 3.5 * blend(t, 4.0, 5.5)

    # ------------------------------------------------
    # Maneuver 2:
    # climb
    # ------------------------------------------------

    z += 3.2 * blend(t, 5.0, 7.0)

    # descend
    z -= 3.2 * blend(t, 7.5, 9.0)

    # ------------------------------------------------
    # Maneuver 3:
    # strong right dodge
    # ------------------------------------------------

    x += 4.2 * blend(t, 9.0, 11.0)

    x -= 4.2 * blend(t, 11.0, 12.5)

    # ------------------------------------------------
    # Maneuver 4:
    # dive then recover
    # ------------------------------------------------

    z -= 2.1 * blend(t, 12.0, 13.5)

    z += 2.1 * blend(t, 13.5, 15.0)

    # ------------------------------------------------
    # Maneuver 5:
    # S-turn
    # ------------------------------------------------

    if 14.5 < t < 18.5:

        phase = (
            t - 14.5
        ) / 4.0

        x += (
            2.2
            * math.sin(
                phase
                * math.pi
                * 2
            )
        )

    return Vector((x, y, z))


# ============================================================
# ESTIMATE VELOCITY
# ============================================================

def trajectory_velocity(t):

    dt = 0.01

    p0 = center_trajectory(max(0, t - dt))
    p1 = center_trajectory(min(DURATION, t + dt))

    return (p1 - p0) / (2 * dt)


# ============================================================
# COMPUTE AIRCRAFT ORIENTATION
#
# yaw   -> path direction
# pitch -> climb/descent
# roll  -> lateral turning
# ============================================================

def trajectory_orientation(t):

    v = trajectory_velocity(t)

    vx = v.x
    vy = v.y
    vz = v.z

    yaw = math.atan2(
        -vx,
        vy
    )

    horizontal_speed = math.sqrt(
        vx * vx + vy * vy
    )

    pitch = math.atan2(
        vz,
        horizontal_speed
    )

    # approximate lateral acceleration for bank command
    dt = 0.06

    vm = trajectory_velocity(max(0, t - dt))
    vp = trajectory_velocity(min(DURATION, t + dt))

    acceleration = (
        vp - vm
    ) / (2 * dt)

    lateral_acc = acceleration.x

    # exaggerated for visual clarity
    roll = -math.atan2(
        lateral_acc,
        9.81
    ) * 2.0

    roll = max(
        math.radians(-45),
        min(
            math.radians(45),
            roll
        )
    )

    # Aircraft created along +Y.
    return Euler(
        (
            pitch,
            roll,
            yaw
        ),
        'XYZ'
    )


# ============================================================
# ANIMATE AIRCRAFT
# ============================================================

for frame in range(
    scene.frame_start,
    scene.frame_end + 1
):

    t = (
        frame - 1
    ) / FPS

    center = center_trajectory(t)

    orientation = trajectory_orientation(t)

    # convert orientation into matrix
    R = orientation.to_matrix()

    for jet, local_offset in zip(
        jets,
        formation_offsets
    ):

        # Rotate formation offsets with heading
        rotated_offset = (
            R @ local_offset
        )

        # Keep formation approximately planar while banking.
        position = (
            center
            + rotated_offset
        )

        jet.location = position
        jet.rotation_euler = orientation

        jet.keyframe_insert(
            data_path="location",
            frame=frame
        )

        jet.keyframe_insert(
            data_path="rotation_euler",
            frame=frame
        )


# ============================================================
# CAMERA
# ============================================================

bpy.ops.object.camera_add()

camera = bpy.context.object
camera.name = "ChaseCamera"

scene.camera = camera

camera.data.lens = 42

# ============================================================
# CAMERA ANIMATION
# ============================================================

for frame in range(
    scene.frame_start,
    scene.frame_end + 1
):

    t = (
        frame - 1
    ) / FPS

    center = center_trajectory(t)

    velocity = trajectory_velocity(t)
    forward = velocity.normalized()

    # camera is behind and above formation
    camera_pos = (
        center
        - forward * 16.0
        + Vector((0, 0, 9.0))
    )

    camera.location = camera_pos

    target = (
        center
        + forward * 8.0
    )

    direction = (
        target
        - camera_pos
    )

    camera.rotation_euler = direction.to_track_quat(
        '-Z',
        'Y'
    ).to_euler()

    camera.keyframe_insert(
        data_path="location",
        frame=frame
    )

    camera.keyframe_insert(
        data_path="rotation_euler",
        frame=frame
    )


# ============================================================
# SUN LIGHT
# ============================================================

bpy.ops.object.light_add(
    type='SUN',
    location=(0, 30, 25)
)

sun = bpy.context.object
sun.name = "Sun"

sun.rotation_euler = (
    math.radians(35),
    0,
    math.radians(-35)
)

sun.data.energy = 3.0


# ============================================================
# AREA LIGHT
# ============================================================

bpy.ops.object.light_add(
    type='AREA',
    location=(0, 15, 15)
)

area = bpy.context.object
area.data.energy = 1400
area.data.shape = 'RECTANGLE'
area.data.size = 20


# ============================================================
# SUNSET BACKGROUND
# ============================================================

world = scene.world

world.use_nodes = True

bg = world.node_tree.nodes.get(
    "Background"
)

bg.inputs["Color"].default_value = (
    0.10,
    0.035,
    0.02,
    1
)

bg.inputs["Strength"].default_value = 0.35


# ============================================================
# INTERPOLATION
# ============================================================

for obj in jets + [camera]:

    if obj.animation_data is None:
        continue

    action = obj.animation_data.action

    if action is None:
        continue

    for fcurve in action.fcurves:

        for point in fcurve.keyframe_points:
            point.interpolation = 'BEZIER'


# ============================================================
# SAVE BLEND
# ============================================================

blend_path = bpy.path.abspath(
    "//f16_formation_animation.blend"
)

bpy.ops.wm.save_as_mainfile(
    filepath=blend_path
)

print("=" * 60)
print("Scene generated.")
print("Frames:", scene.frame_start, "-", scene.frame_end)
print("FPS:", FPS)
print("Blend:", blend_path)
print("Video:", bpy.path.abspath(OUTPUT_FILE))
print("=" * 60)

# ============================================================
# OPTIONAL AUTO RENDER
# Uncomment to immediately render the entire animation.
# ============================================================

bpy.ops.render.render(animation=True)