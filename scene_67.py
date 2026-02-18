import os
import time
import math
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))
TM_PORT = int(os.getenv("CARLA_TM_PORT", 8000))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

DURATION_SECONDS = 15.0

# --- Crossing placement relative to ego start ---
CROSSING_AHEAD_M = 75.0
STOPLINE_BEFORE_TRACKS_M = 12.0

# --- Visuals ---
TRACKS_HALF_WIDTH_M = 7.5
TRACK_GAUGE_OFFSET_M = 0.8
TIE_HALF_LENGTH_M = 2.6
TIE_SPACING_M = 0.7
TIES_COUNT_EACH_SIDE = 10

SIGN_BEFORE_STOPLINE_M = 10.0

# --- Warning logic ---
DECISION_ZONE_START_M = 45.0   # warnings begin when ego is within this distance to stopline
GATE_FULLY_DOWN_AFTER = 2.8    # seconds after warning starts

# --- Warning lights and barriers positions ---
LIGHT_POST_BEFORE_TRACKS_M = 7.0
LIGHT_POST_LATERAL_M = 6.5

GATE_PIVOT_LATERAL_M = 4.2
GATE_ARM_LENGTH_M = 8.0

# --- Train-like crossing ---
TRAIN_START_OFFSET_M = 55.0
TRAIN_SPEED_MPS = 13.0
TRAIN_CROSSING_WINDOW = 8.5
TRAIN_START_DELAY_AFTER_WARNING = 0.9

# Make debug geometry persistent enough to actually be visible in recordings
DEBUG_DRAW_ALWAYS = True
DEBUG_DRAW_LIFETIME = 0.25

def setup_world(client: carla.Client) -> carla.World:
    world = client.get_world()
    try:
        current_map_name = world.get_map().name
    except Exception:
        current_map_name = ""

    if MAP_NAME not in current_map_name:
        world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    try:
        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world

def vec2(v: carla.Vector3D) -> carla.Vector3D:
    return carla.Vector3D(v.x, v.y, 0.0)

def unit2(v: carla.Vector3D) -> carla.Vector3D:
    n = math.hypot(v.x, v.y)
    if n < 1e-6:
        return carla.Vector3D(1.0, 0.0, 0.0)
    return carla.Vector3D(v.x / n, v.y / n, 0.0)

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def pick_vehicle_bp(bp_lib: carla.BlueprintLibrary, patterns):
    for patt in patterns:
        bps = bp_lib.filter(patt)
        if bps:
            return bps[0]
    return bp_lib.filter("vehicle.*")[0]

def set_vehicle_lights(vehicle: carla.Vehicle, mask: int) -> None:
    try:
        vehicle.set_light_state(carla.VehicleLightState(mask))
    except Exception:
        pass

def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = setup_world(client)
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    debug = world.debug

    actors_to_destroy = []

    # ---------- Spawn Ego ----------
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on map")

    ego_spawn_idx = 31 if len(spawn_points) > 31 else 0
    ego_spawn = spawn_points[ego_spawn_idx]

    ego_bp = pick_vehicle_bp(bp_lib, ["vehicle.tesla.model3", "vehicle.*"])
    try:
        ego_bp.set_attribute("role_name", "hero")
    except Exception:
        pass

    ego = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego is None:
        ego = world.spawn_actor(ego_bp, ego_spawn)
    actors_to_destroy.append(ego)
    world.tick()

    # ---------- PCLA ----------
    route_xml = "./sample_route.xml"
    pcla = PCLA("carl_carlv11", ego, route_xml, client)

    # Encourage moderate speed; keep rule-following enabled
    try:
        tm = client.get_trafficmanager(TM_PORT)
        tm.vehicle_percentage_speed_difference(ego, -5.0)
        tm.ignore_lights_percentage(ego, 0.0)
        tm.ignore_signs_percentage(ego, 0.0)
        tm.update_vehicle_lights(ego, False)
    except Exception:
        tm = None

    # ---------- Spectator / Camera ----------
    spectator = world.get_spectator()

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)
    actors_to_destroy.append(camera)

    image_queue: Queue = Queue()

    def camera_callback(image: carla.Image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_queue.put(array[:, :, :3])

    camera.listen(camera_callback)

    video = cv2.VideoWriter(
        VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (IMAGE_W, IMAGE_H),
    )

    # ---------- Compute crossing reference from ego start, snapped to road ----------
    ego_start_tf = ego.get_transform()

    forward0 = unit2(vec2(ego_start_tf.get_forward_vector()))
    right0 = unit2(vec2(ego_start_tf.get_right_vector()))

    approx_tracks_center = ego_start_tf.location + carla.Location(
        x=forward0.x * CROSSING_AHEAD_M,
        y=forward0.y * CROSSING_AHEAD_M,
        z=0.0,
    )

    tracks_wp = carla_map.get_waypoint(
        approx_tracks_center,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if tracks_wp is not None:
        tracks_center = tracks_wp.transform.location
        road_yaw = tracks_wp.transform.rotation.yaw
        road_fwd = unit2(vec2(tracks_wp.transform.get_forward_vector()))
        road_right = unit2(vec2(tracks_wp.transform.get_right_vector()))
    else:
        tracks_center = approx_tracks_center
        road_yaw = ego_start_tf.rotation.yaw
        road_fwd = forward0
        road_right = right0

    # Track direction runs across the road
    track_dir = road_right

    stopline_loc = tracks_center - carla.Location(
        x=road_fwd.x * STOPLINE_BEFORE_TRACKS_M,
        y=road_fwd.y * STOPLINE_BEFORE_TRACKS_M,
        z=0.0,
    )
    stopline_wp = carla_map.get_waypoint(
        stopline_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    if stopline_wp is not None:
        stopline_loc = stopline_wp.transform.location

    # ---------- Debug-drawn railroad crossing visuals ----------
    def draw_crossing_visuals(life_time: float):
        if not DEBUG_DRAW_ALWAYS:
            return

        z = 0.06
        rail_color = carla.Color(200, 200, 200)
        tie_color = carla.Color(110, 75, 40)
        stop_color = carla.Color(255, 255, 255)
        warn_color = carla.Color(255, 40, 40)
        sign_color = carla.Color(245, 245, 245)

        # Rails (two parallel lines across road)
        for offset in [-TRACK_GAUGE_OFFSET_M, TRACK_GAUGE_OFFSET_M]:
            p0 = tracks_center + carla.Location(z=z) + carla.Location(
                x=road_fwd.x * offset,
                y=road_fwd.y * offset,
                z=0.0,
            ) - carla.Location(
                x=track_dir.x * TRACKS_HALF_WIDTH_M,
                y=track_dir.y * TRACKS_HALF_WIDTH_M,
                z=0.0,
            )
            p1 = tracks_center + carla.Location(z=z) + carla.Location(
                x=road_fwd.x * offset,
                y=road_fwd.y * offset,
                z=0.0,
            ) + carla.Location(
                x=track_dir.x * TRACKS_HALF_WIDTH_M,
                y=track_dir.y * TRACKS_HALF_WIDTH_M,
                z=0.0,
            )
            debug.draw_line(p0, p1, thickness=0.14, color=rail_color, life_time=life_time)

        # Ties (perpendicular to rails, along track dir)
        for k in range(-TIES_COUNT_EACH_SIDE, TIES_COUNT_EACH_SIDE + 1):
            along = k * TIE_SPACING_M
            tie_center = tracks_center + carla.Location(
                x=track_dir.x * along,
                y=track_dir.y * along,
                z=z - 0.02,
            )
            a = tie_center - carla.Location(x=road_fwd.x * TIE_HALF_LENGTH_M, y=road_fwd.y * TIE_HALF_LENGTH_M, z=0.0)
            b = tie_center + carla.Location(x=road_fwd.x * TIE_HALF_LENGTH_M, y=road_fwd.y * TIE_HALF_LENGTH_M, z=0.0)
            debug.draw_line(a, b, thickness=0.06, color=tie_color, life_time=life_time)

        # Stop line at decision point
        sl_a = stopline_loc - carla.Location(x=road_right.x * 2.4, y=road_right.y * 2.4, z=0.0) + carla.Location(z=z)
        sl_b = stopline_loc + carla.Location(x=road_right.x * 2.4, y=road_right.y * 2.4, z=0.0) + carla.Location(z=z)
        debug.draw_line(sl_a, sl_b, thickness=0.22, color=stop_color, life_time=life_time)

        # "RAILROAD" text hint (server-side only, but helps in CARLA render review)
        sign_loc = stopline_loc - carla.Location(x=road_fwd.x * SIGN_BEFORE_STOPLINE_M, y=road_fwd.y * SIGN_BEFORE_STOPLINE_M, z=0.0) + carla.Location(z=1.6)
        debug.draw_string(sign_loc, "RAILROAD CROSSING", draw_shadow=True, color=sign_color, life_time=life_time)

        # Add a red "WARNING" string near tracks when active (set outside based on state)
        return warn_color

    # ---------- Spawn warning "signal posts" (static vehicles with hazards) ----------
    warning_posts = []
    warning_active = False
    warning_start_time = None

    def spawn_warning_post(side_sign: float):
        loc = tracks_center - carla.Location(
            x=road_fwd.x * LIGHT_POST_BEFORE_TRACKS_M,
            y=road_fwd.y * LIGHT_POST_BEFORE_TRACKS_M,
            z=0.0,
        ) + carla.Location(
            x=road_right.x * LIGHT_POST_LATERAL_M * side_sign,
            y=road_right.y * LIGHT_POST_LATERAL_M * side_sign,
            z=0.0,
        )
        loc.z += 0.25
        rot = carla.Rotation(yaw=road_yaw, pitch=0.0, roll=0.0)
        tf = carla.Transform(loc, rot)

        bp = pick_vehicle_bp(bp_lib, ["vehicle.micro.microlino", "vehicle.citroen.c3", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_warning_post")
        except Exception:
            pass

        v = world.try_spawn_actor(bp, tf)
        if v is None:
            return None
        try:
            v.set_simulate_physics(False)
            v.set_autopilot(False)
        except Exception:
            pass
        set_vehicle_lights(v, int(carla.VehicleLightState.Position))
        return v

    post_left = spawn_warning_post(+1.0)
    post_right = spawn_warning_post(-1.0)
    if post_left:
        warning_posts.append(post_left)
        actors_to_destroy.append(post_left)
    if post_right:
        warning_posts.append(post_right)
        actors_to_destroy.append(post_right)

    # ---------- Spawn barrier gate arm (static long vehicle animated down) ----------
    gate_vehicle = None
    gate_up_tf = None
    gate_down_tf = None
    gate_down = False

    def spawn_gate():
        nonlocal gate_vehicle, gate_up_tf, gate_down_tf

        pivot_loc = stopline_loc + carla.Location(
            x=road_right.x * GATE_PIVOT_LATERAL_M,
            y=road_right.y * GATE_PIVOT_LATERAL_M,
            z=0.0,
        )
        pivot_loc.z += 0.4

        bp = pick_vehicle_bp(bp_lib, ["vehicle.ford.ambulance", "vehicle.carlamotors.carlacola", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_gate_arm")
        except Exception:
            pass

        # Down blocks lane across road; Up rests along shoulder (angled away)
        yaw_down = road_yaw + 90.0
        yaw_up = yaw_down - 80.0

        arm_center_down = pivot_loc + carla.Location(
            x=-road_right.x * (GATE_ARM_LENGTH_M * 0.40),
            y=-road_right.y * (GATE_ARM_LENGTH_M * 0.40),
            z=0.0,
        )

        arm_center_up = pivot_loc + carla.Location(
            x=-road_right.x * (GATE_ARM_LENGTH_M * 0.65),
            y=-road_right.y * (GATE_ARM_LENGTH_M * 0.65),
            z=0.9,
        )

        gate_down_tf = carla.Transform(arm_center_down, carla.Rotation(yaw=yaw_down, pitch=0.0, roll=0.0))
        gate_up_tf = carla.Transform(arm_center_up, carla.Rotation(yaw=yaw_up, pitch=0.0, roll=0.0))

        gate_vehicle = world.try_spawn_actor(bp, gate_up_tf)
        if gate_vehicle is None:
            return

        actors_to_destroy.append(gate_vehicle)
        try:
            gate_vehicle.set_simulate_physics(False)
            gate_vehicle.set_autopilot(False)
        except Exception:
            pass
        set_vehicle_lights(gate_vehicle, int(carla.VehicleLightState.Position))

    spawn_gate()

    # ---------- Spawn train-like actor that crosses at the tracks ----------
    train = None
    train_active = False
    train_start_time = None

    def spawn_train_like_actor():
        nonlocal train
        bp = pick_vehicle_bp(bp_lib, ["vehicle.bus.*", "vehicle.carlamotors.carlacola", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_train_like")
        except Exception:
            pass

        start_loc = tracks_center - carla.Location(
            x=track_dir.x * TRAIN_START_OFFSET_M,
            y=track_dir.y * TRAIN_START_OFFSET_M,
            z=0.0,
        )
        start_loc.z += 0.35
        yaw = math.degrees(math.atan2(track_dir.y, track_dir.x))
        tf = carla.Transform(start_loc, carla.Rotation(yaw=yaw))

        train = world.try_spawn_actor(bp, tf)
        if train is None:
            return

        actors_to_destroy.append(train)
        try:
            train.set_autopilot(False)
            train.set_simulate_physics(True)
        except Exception:
            pass

        # Lights on to be visible
        set_vehicle_lights(train, int(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

    spawn_train_like_actor()

    start_wall = time.time()

    try:
        while True:
            wall_elapsed = time.time() - start_wall
            if wall_elapsed >= DURATION_SECONDS:
                break

            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds

            # --- Ego control from PCLA ---
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            dist_to_stopline = ego_loc.distance(stopline_loc)

            # --- Trigger warnings near decision zone ---
            if warning_start_time is None and dist_to_stopline < DECISION_ZONE_START_M:
                warning_start_time = sim_time
                warning_active = True
                train_start_time = sim_time + TRAIN_START_DELAY_AFTER_WARNING

            # --- Warning lights: hazards on both posts ---
            if warning_active:
                for post in warning_posts:
                    if post is None:
                        continue
                    set_vehicle_lights(
                        post,
                        int(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LeftBlinker
                            | carla.VehicleLightState.RightBlinker
                        ),
                    )

            # --- Gate animation: interpolate to down position ---
            if gate_vehicle is not None and warning_start_time is not None and not gate_down:
                t = (sim_time - warning_start_time) / max(GATE_FULLY_DOWN_AFTER, 1e-3)
                t = clamp(t, 0.0, 1.0)

                loc = carla.Location(
                    x=gate_up_tf.location.x * (1 - t) + gate_down_tf.location.x * t,
                    y=gate_up_tf.location.y * (1 - t) + gate_down_tf.location.y * t,
                    z=gate_up_tf.location.z * (1 - t) + gate_down_tf.location.z * t,
                )
                yaw = gate_up_tf.rotation.yaw * (1 - t) + gate_down_tf.rotation.yaw * t
                tf = carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0))

                try:
                    gate_vehicle.set_transform(tf)
                except Exception:
                    pass

                if t >= 0.999:
                    gate_down = True

            # --- Train movement across the tracks (reinforces rail decision point) ---
            if train is not None and train_start_time is not None and sim_time >= train_start_time:
                if not train_active:
                    train_active = True
                if sim_time - train_start_time <= TRAIN_CROSSING_WINDOW:
                    try:
                        train.set_target_velocity(track_dir * TRAIN_SPEED_MPS)
                    except Exception:
                        pass
                else:
                    try:
                        train.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    except Exception:
                        pass

            # --- Draw crossing visuals every frame (persistent enough to see) ---
            warn_color = draw_crossing_visuals(life_time=DEBUG_DRAW_LIFETIME)
            if warning_active:
                debug.draw_string(tracks_center + carla.Location(z=1.8), "WARNING: TRAIN", draw_shadow=True, color=warn_color, life_time=DEBUG_DRAW_LIFETIME)

            # --- Chase camera: slightly above/behind ego, forward-facing aligned with ego ---
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_loc - forward * 7.0 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-10.0, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # --- Tick & record ---
            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        try:
            if camera:
                camera.stop()
        except Exception:
            pass

        try:
            video.release()
        except Exception:
            pass

        try:
            pcla.cleanup()
        except Exception:
            pass

        for a in reversed(actors_to_destroy):
            safe_destroy(a)

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        try:
            tm = client.get_trafficmanager(TM_PORT)
            tm.set_synchronous_mode(False)
        except Exception:
            pass

if __name__ == "__main__":
    main()