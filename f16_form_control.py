import bpy
import math
import os
from mathutils import Vector, Euler


# ============================================================
# SETTINGS
# ============================================================

FPS = 30
DURATION = 20.0#20.0
N_FRAMES = int(FPS * DURATION)

PROJECT_DIR = "/"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "assets",
    "f16.glb"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "f16_formation_corridor_new.mp4"
)

BLEND_FILE = os.path.join(
    PROJECT_DIR,
    "f16_formation_animation.blend"
)

# Adjust if needed after inspecting the imported model
MODEL_SCALE = 0.12

MODEL_ROTATION = (
    math.radians(0),
    math.radians(0),
    math.radians(0),
)


# ============================================================
# CORRIDOR
# ============================================================

CORRIDOR_LENGTH = 110.0
#CORRIDOR_WIDTH = 16.0
CORRIDOR_WIDTH = 18.0
CORRIDOR_HEIGHT = 10.0

# FORMATION_WIDTH = 2.6
# FORMATION_LENGTH = 2.8

FORMATION_WIDTH = 2.2
FORMATION_LENGTH = 2.4


# ============================================================
# RESET SCENE
# ============================================================

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene

# Blender 3.0.1
scene.render.engine = "BLENDER_EEVEE"

scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = N_FRAMES

scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
scene.render.ffmpeg.ffmpeg_preset = "GOOD"

scene.render.filepath = OUTPUT_FILE

# Better ambient shading / Eevee quality
scene.eevee.use_gtao = True
scene.eevee.gtao_distance = 5.0
scene.eevee.gtao_factor = 1.5
scene.eevee.use_soft_shadows = True
scene.eevee.taa_render_samples = 96


# ============================================================
# MATERIALS
# ============================================================

def make_material(name, color, roughness=0.8, metallic=0.0):

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    bsdf.inputs["Base Color"].default_value = (
        color[0],
        color[1],
        color[2],
        1.0
    )

    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic

    return mat


# Darker, higher-contrast simulation style
MAT_FLOOR = make_material(
    "FloorMaterial",
    (0.055, 0.22, 0.09),
    roughness=0.82
)

MAT_WALL = make_material(
    "WallMaterial",
    (0.045, 0.16, 0.07),
    roughness=0.92
)

MAT_OBSTACLE = make_material(
    "ObstacleMaterial",
    (0.22, 0.24, 0.18),
    roughness=0.78
)


# ============================================================
# GEOMETRY HELPER
# ============================================================

def create_cube(name, location, scale, material):

    bpy.ops.mesh.primitive_cube_add(
        location=location
    )

    obj = bpy.context.object
    obj.name = name
    obj.scale = scale

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True
    )

    obj.data.materials.append(material)

    return obj


# ============================================================
# FLOOR
# ============================================================

create_cube(
    "Ground",
    (
        0,
        CORRIDOR_LENGTH / 2,
        -0.2
    ),
    (
        CORRIDOR_WIDTH / 2,
        CORRIDOR_LENGTH / 2,
        0.2
    ),
    MAT_FLOOR
)


# ============================================================
# CORRIDOR WALLS
# ============================================================

create_cube(
    "LeftWall",
    (
        -CORRIDOR_WIDTH / 2 - 0.2,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    (
        0.2,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    MAT_WALL
)

create_cube(
    "RightWall",
    (
        CORRIDOR_WIDTH / 2 + 0.2,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    (
        0.2,
        CORRIDOR_LENGTH / 2,
        CORRIDOR_HEIGHT / 2
    ),
    MAT_WALL
)


# ============================================================
# OBSTACLES
# ============================================================

obstacles = [
    (-3.7, 20, 1.5, 1.4, 2.0, 1.5),
    ( 3.6, 32, 2.2, 1.4, 2.0, 2.2),
    ( 0.0, 44, 1.8, 1.5, 2.0, 1.8),
    (-3.8, 57, 2.5, 1.4, 2.0, 2.5),
    ( 3.7, 70, 1.8, 1.4, 2.0, 1.8),
    ( 0.0, 83, 2.4, 1.5, 2.0, 2.4),
]

# obstacles = [
#
#     (-3.7, 20, 1.8, 1.5, 2.0, 1.8),
#
#     (3.6, 32, 3.0, 1.5, 2.0, 3.0),
#
#     (0.0, 44, 2.0, 1.8, 2.0, 2.0),
#
#     (-3.8, 57, 3.8, 1.6, 2.0, 3.8),
#
#     (3.7, 70, 2.4, 1.6, 2.0, 2.4),
#
#     (0.0, 83, 3.8, 1.8, 2.0, 3.8),
# ]

for i, obs in enumerate(obstacles):

    x, y, z, sx, sy, sz = obs

    create_cube(
        "Obstacle_%02d" % i,
        (x, y, z),
        (sx, sy, sz),
        MAT_OBSTACLE
    )


# ============================================================
# IMPORT F-16 MODEL
# ============================================================

def import_f16(name):

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "F-16 model not found:\n%s" % MODEL_PATH
        )

    # --------------------------------------------------------
    # Motion root
    # --------------------------------------------------------

    motion_root = bpy.data.objects.new(
        name + "_motion",
        None
    )

    bpy.context.collection.objects.link(
        motion_root
    )

    # --------------------------------------------------------
    # Model root
    # --------------------------------------------------------

    model_root = bpy.data.objects.new(
        name + "_model",
        None
    )

    bpy.context.collection.objects.link(
        model_root
    )

    model_root.parent = motion_root

    objects_before = set(bpy.data.objects)

    bpy.ops.import_scene.gltf(
        filepath=MODEL_PATH
    )

    objects_after = set(bpy.data.objects)

    imported_objects = list(
        objects_after - objects_before
    )

    imported_set = set(imported_objects)

    # Parent only top-level imported objects
    for obj in imported_objects:

        if obj.parent not in imported_set:
            obj.parent = model_root

    model_root.scale = (
        MODEL_SCALE,
        MODEL_SCALE,
        MODEL_SCALE
    )

    model_root.rotation_euler = MODEL_ROTATION

    # Make imported F-16 materials respond more clearly to directional light
    for obj in imported_objects:
        if obj.type != "MESH":
            continue
        for mat in obj.data.materials:
            if mat is None or not mat.use_nodes:
                continue
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Roughness"].default_value = 0.38
                if "Specular" in bsdf.inputs:
                    bsdf.inputs["Specular"].default_value = 0.55

    return motion_root


# ============================================================
# CREATE F-16 FORMATION
# ============================================================

jets = []

for i in range(4):

    jets.append(
        import_f16(
            "F16_%02d" % (i + 1)
        )
    )


# ============================================================
# DIAMOND FORMATION
# ============================================================

formation_offsets = [

    # Leader
    Vector((
        0.0,
        FORMATION_LENGTH,
        0.0
    )),

    # Left
    Vector((
        -FORMATION_WIDTH,
        0.0,
        0.0
    )),

    # Right
    Vector((
        FORMATION_WIDTH,
        0.0,
        0.0
    )),

    # Rear
    Vector((
        0.0,
        -FORMATION_LENGTH,
        0.0
    )),
]


# ============================================================
# SMOOTH MOTION
# ============================================================

def smoothstep(x):

    x = max(
        0.0,
        min(1.0, x)
    )

    return (
        x * x *
        (3.0 - 2.0 * x)
    )


def transition(t, t0, t1):

    if t <= t0:
        return 0.0

    if t >= t1:
        return 1.0

    return smoothstep(
        (t - t0) /
        (t1 - t0)
    )


# ============================================================
# FORMATION TRAJECTORY
# ============================================================

def center_trajectory_(t):

    # Forward motion
    y = 6.0 + 4.5 * t

    x = 0.0
    z = 3.8

    # --------------------------------------------------------
    # Obstacle 1 near y = 20
    # Dodge RIGHT
    # --------------------------------------------------------
    x += 4.8 * transition(t, 2.0, 3.3)
    x -= 4.8 * transition(t, 3.8, 5.0)

    # --------------------------------------------------------
    # Obstacle 2 near y = 32
    # Climb
    # --------------------------------------------------------
    z += 3.5 * transition(t, 5.0, 6.3)
    z -= 3.5 * transition(t, 7.0, 8.2)

    # z += 4.0 * transition(t, 4.0, 5.3)
    # z -= 4.0 * transition(t, 6.5, 7.8)

    # --------------------------------------------------------
    # Obstacle 3 near y = 44
    # Dodge LEFT
    # --------------------------------------------------------
    x -= 5.0 * transition(t, 8.0, 9.2)
    x += 5.0 * transition(t, 9.8, 11.0)

    # --------------------------------------------------------
    # Obstacle 4 near y = 57
    # Climb high
    # --------------------------------------------------------
    # z += 4.2 * transition(t, 10.8, 12.0)
    # z -= 4.2 * transition(t, 12.8, 14.0)

    z += 4.0 * transition(t, 4.0, 5.3)
    z -= 4.0 * transition(t, 6.5, 7.8)

    # --------------------------------------------------------
    # Obstacle 5 near y = 70
    # Dodge RIGHT
    # --------------------------------------------------------
    x += 4.8 * transition(t, 14.0, 15.2)
    x -= 4.8 * transition(t, 15.8, 17.0)

    # Obstacle is on RIGHT → dodge LEFT
    x -= 4.5 * transition(t, 13.0, 14.0)
    x += 4.5 * transition(t, 14.8, 15.8)

    # --------------------------------------------------------
    # Obstacle 6 near y = 83
    # Left dodge + small climb
    # --------------------------------------------------------
    # x -= 4.5 * transition(t, 16.5, 17.5)
    # z += 2.5 * transition(t, 16.5, 17.5)
    #
    # x += 4.5 * transition(t, 18.0, 19.0)
    # z -= 2.5 * transition(t, 18.0, 19.0)

    x -= 4.0 * transition(t, 15.5, 16.5)
    z += 3.0 * transition(t, 15.5, 16.5)

    x += 4.0 * transition(t, 17.8, 19.0)
    z -= 3.0 * transition(t, 17.8, 19.0)

    return Vector((x, y, z))

def center_trajectory(t):

    y = 6.0 + 4.5 * t

    x = 0.0
    z = 3.8

    # Obstacle 1: left obstacle -> dodge right
    x += 4.5 * transition(t, 1.8, 2.8)
    x -= 4.5 * transition(t, 3.5, 4.5)

    # Obstacle 2: climb, but retain original height
    z += 2.7 * transition(t, 4.0, 5.2)
    z -= 2.7 * transition(t, 6.5, 7.8)

    # Obstacle 3: central -> dodge left
    x -= 4.5 * transition(t, 7.2, 8.2)
    x += 4.5 * transition(t, 9.0, 10.0)

    # Obstacle 4: start climb earlier, same height
    z += 2.7 * transition(t, 9.6, 10.8)
    z -= 2.7 * transition(t, 12.0, 13.0)

    # Obstacle 5: right obstacle -> dodge left
    x -= 4.5 * transition(t, 12.8, 13.8)
    x += 4.5 * transition(t, 14.8, 15.8)

    # Obstacle 6: mostly lateral avoidance
    x -= 4.5 * transition(t, 15.4, 16.4)
    z += 1.5 * transition(t, 15.4, 16.4)

    x += 4.5 * transition(t, 17.8, 19.0)
    z -= 1.5 * transition(t, 17.8, 19.0)

    return Vector((x, y, z))


# ============================================================
# VELOCITY
# ============================================================

def trajectory_velocity(t):

    dt = 0.015

    t0 = max(
        0.0,
        t - dt
    )

    t1 = min(
        DURATION,
        t + dt
    )

    p0 = center_trajectory(t0)
    p1 = center_trajectory(t1)

    if abs(t1 - t0) < 1e-8:
        return Vector((0, 1, 0))

    return (
        p1 - p0
    ) / (t1 - t0)


# ============================================================
# ACCELERATION
# ============================================================

def trajectory_acceleration(t):

    dt = 0.06

    vm = trajectory_velocity(
        max(0.0, t - dt)
    )

    vp = trajectory_velocity(
        min(DURATION, t + dt)
    )

    return (
        vp - vm
    ) / (2.0 * dt)


# ============================================================
# AIRCRAFT ATTITUDE
# ============================================================

def trajectory_orientation(t):

    velocity = trajectory_velocity(t)

    vx = velocity.x
    vy = velocity.y
    vz = velocity.z

    horizontal_speed = math.sqrt(
        vx * vx +
        vy * vy
    )

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    yaw = math.atan2(
        -vx,
        vy
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    pitch = math.atan2(
        vz,
        horizontal_speed
    )

    # Slight exaggeration for visibility
    pitch *= 1.4

    # --------------------------------------------------------
    # Roll
    # --------------------------------------------------------

    acc = trajectory_acceleration(t)

    lateral_acc = acc.x

    roll = -math.atan2(
        lateral_acc,
        9.81
    )

    # Make banking clearly visible
    roll *= 2.4

    roll = max(
        math.radians(-42),
        min(
            math.radians(42),
            roll
        )
    )

    return Euler(
        (
            pitch,
            roll,
            yaw
        ),
        "XYZ"
    )


# ============================================================
# ANIMATE FORMATION
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

    # Use yaw for formation geometry
    yaw_rotation = Euler(
        (
            0.0,
            0.0,
            orientation.z
        ),
        "XYZ"
    ).to_matrix()

    for jet, offset in zip(
        jets,
        formation_offsets
    ):

        rotated_offset = (
            yaw_rotation @ offset
        )

        jet.location = (
            center +
            rotated_offset
        )

        jet.rotation_mode = "XYZ"
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
#
# High rear-left oblique camera similar to your reference.
# ============================================================

bpy.ops.object.camera_add()

camera = bpy.context.object
camera.name = "Camera"

scene.camera = camera

camera.data.lens = 58


for frame in range(
    scene.frame_start,
    scene.frame_end + 1
):

    t = (
        frame - 1
    ) / FPS

    center = center_trajectory(t)

    velocity = trajectory_velocity(t)

    if velocity.length < 1e-5:

        forward = Vector(
            (0, 1, 0)
        )

    else:

        forward = velocity.normalized()


    camera_position = Vector((
        0.0,             # always centered laterally
        center.y - 11.5, # closer chase view
        8.0              # lower camera height
    ))

    # camera_position = (
    #     center
    #     - forward * 17.0
    #     + Vector(
    #         (
    #             -8.5,
    #             0.0,
    #             13.0
    #         )
    #     )
    # )

    camera.location = camera_position

    # Look ahead of center
    # target = (
    #     center
    #     + forward * 6.0
    #     + Vector(
    #         (
    #             0.0,
    #             0.0,
    #             0.3
    #         )
    #     )
    # )

    target = Vector((
        center.x,
        center.y + 4.5,
        center.z + 0.3
    ))

    direction = (
        target -
        camera_position
    )

    camera.rotation_euler = (
        direction
        .to_track_quat(
            "-Z",
            "Y"
        )
        .to_euler()
    )

    camera.keyframe_insert(
        data_path="location",
        frame=frame
    )

    camera.keyframe_insert(
        data_path="rotation_euler",
        frame=frame
    )


# ============================================================
# LIGHTING — strong directional key light
# ============================================================

bpy.ops.object.light_add(
    type="SUN",
    location=(0, 0, 20)
)

sun = bpy.context.object
sun.name = "Sun"

sun.rotation_euler = (
    math.radians(35),
    math.radians(-25),
    math.radians(-40)
)

sun.data.energy = 4.0
sun.data.angle = math.radians(8.0)


# Weak fill light: preserves detail without washing out shadows
bpy.ops.object.light_add(
    type="AREA",
    location=(-5, 10, 12)
)

area = bpy.context.object
area.name = "FillLight"
area.data.energy = 120
area.data.size = 12


# ============================================================
# WORLD BACKGROUND — dark green ambient illumination
# ============================================================

scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get("Background")

background.inputs["Color"].default_value = (
    0.008,
    0.018,
    0.009,
    1.0
)
background.inputs["Strength"].default_value = 0.18


# ============================================================
# COLOR MANAGEMENT
# ============================================================

# Blender 3.0.1-safe baseline. Avoid Filmic because this installation
# previously reported that Filmic is unavailable.
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = -0.3
scene.view_settings.gamma = 1.0

# ============================================================
# SMOOTH INTERPOLATION
# ============================================================

for obj in jets + [camera]:

    if obj.animation_data is None:
        continue

    if obj.animation_data.action is None:
        continue

    for fcurve in obj.animation_data.action.fcurves:

        for key in fcurve.keyframe_points:

            key.interpolation = "BEZIER"


# ============================================================
# SAVE SCENE
# ============================================================

bpy.ops.wm.save_as_mainfile(
    filepath=BLEND_FILE
)

print("")
print("=" * 70)
print("F-16 Blender scene generated")
print("=" * 70)

print("Model:")
print(MODEL_PATH)

print("")
print("Frames: %d" % N_FRAMES)
print("FPS: %d" % FPS)

print("")
print("Blend:")
print(BLEND_FILE)

print("")
print("Video:")
print(OUTPUT_FILE)

print("=" * 70)


# ============================================================
# RENDER
# ============================================================

print("")
print("Rendering animation...")
print("")

bpy.ops.render.render(
    animation=True
)

print("")
print("=" * 70)
print("RENDER COMPLETE")
print(OUTPUT_FILE)
print("=" * 70)
