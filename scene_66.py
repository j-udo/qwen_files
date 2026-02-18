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

# ===================== CONFIG =====================

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

# Railroad crossing "illusion" + behavior tuning
CROSSING_AHEAD_M = 65.0
DECISION_ZONE_START_M = 40.0  # when warning begins (distance to stopline)
GATE_FULLY_DOWN_AFTER = 2.5   # seconds after warning start

STOPLINE_BEFORE_TRACKS_M = 10.0
TRACKS_HALF_WIDTH_M = 4.0     # visual track span each side from center
TRACKS_COUNT = 2

LIGHT_POST_BEFORE_TRACKS_M = 6.0
LIGHT_POST_LATERAL_M = 5.5

GATE_PIVOT_LATERAL_M = 3.5
GATE_ARM_LENGTH_M = 7.0

TRAIN_START_OFFSET_M = 45.0
TRAIN_SPEED_MPS = 14.0
TRAIN_CROSSING_WINDOW = 9.0   # seconds of train motion after it starts

DEBUG_DRAW = False

# ==================================================


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


def unit2(v: carla.Vector3D) -> carla.Vector3D:
    n = math.hypot(v.x, v.y)
    if n < 1e-6:
        return carla.Vector3D(1.0, 0.0, 0.0)
    return carla.Vector3D(v.x / n, v.y / n, 0.0)


def vec2(v: carla.Vector3D) -> carla.Vector3D:
    return carla.Vector3D(v.x, v.y, 0.0)


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

    # Encourage moderate speed to create a real decision point
    try:
        tm = client.get_trafficmanager(TM_PORT)
        tm.vehicle_percentage_speed_difference(ego, -10.0)  # slightly above limit (moderate)
        tm.ignore_lights_percentage(ego, 0.0)
        tm.ignore_signs_percentage(ego, 0.0)
    except Exception:
        tm = None

    # ---------- Spectator / Camera (rigidly attached to spectator, spectator follows ego) ----------
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

    # ---------- Define crossing reference using ego start pose ----------
    ego_start_tf = ego.get_transform()
    forward0 = unit2(vec2(ego_start_tf.get_forward_vector()))
    right0 = unit2(vec2(ego_start_tf.get_right_vector()))

    approx_tracks_center = ego_start_tf.location + carla.Location(
        x=forward0.x * CROSSING_AHEAD_M,
        y=forward0.y * CROSSING_AHEAD_M,
        z=0.0,
    )
    tracks_wp = carla_map.get_waypoint(approx_tracks_center, project_to_road=True, lane_type=carla.LaneType.Driving)
    if tracks_wp is not None:
        tracks_center = tracks_wp.transform.location
        tracks_heading_yaw = tracks_wp.transform.rotation.yaw
    else:
        tracks_center = approx_tracks_center
        tracks_heading_yaw = ego_start_tf.rotation.yaw

    stopline_loc = tracks_center - carla.Location(
        x=forward0.x * STOPLINE_BEFORE_TRACKS_M,
        y=forward0.y * STOPLINE_BEFORE_TRACKS_M,
        z=0.0,
    )
    stopline_wp = carla_map.get_waypoint(stopline_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if stopline_wp is not None:
        stopline_loc = stopline_wp.transform.location

    # ---------- Visual railroad tracks + crossbuck markers (debug geometry, always visible for some seconds) ----------
    # CARLA default towns may not have a modeled railroad crossing; we render strong visual cues using debug.
    def draw_crossing_visuals(life_time: float = 0.11):
        if not DEBUG_DRAW:
            return

        # Track direction is perpendicular to road direction
        road_yaw = tracks_heading_yaw
        road_fwd = carla.Vector3D(x=math.cos(math.radians(road_yaw)), y=math.sin(math.radians(road_yaw)), z=0.0)
        road_right = carla.Vector3D(x=-road_fwd.y, y=road_fwd.x, z=0.0)

        track_dir = road_right  # run tracks across the road
        rail_color = carla.Color(180, 180, 180)
        tie_color = carla.Color(90, 60, 30)

        for i in range(TRACKS_COUNT):
            offset = (i - (TRACKS_COUNT - 1) / 2.0) * 1.6
            p0 = tracks_center + carla.Location(x=road_fwd.x * offset, y=road_fwd.y * offset, z=0.05) - carla.Location(
                x=track_dir.x * TRACKS_HALF_WIDTH_M, y=track_dir.y * TRACKS_HALF_WIDTH_M, z=0.0
            )
            p1 = tracks_center + carla.Location(x=road_fwd.x * offset, y=road_fwd.y * offset, z=0.05) + carla.Location(
                x=track_dir.x * TRACKS_HALF_WIDTH_M, y=track_dir.y * TRACKS_HALF_WIDTH_M, z=0.0
            )
            debug.draw_line(p0, p1, thickness=0.12, color=rail_color, life_time=life_time)

        # Ties
        for k in range(-6, 7):
            along = k * 0.65
            tie_center = tracks_center + carla.Location(x=track_dir.x * along, y=track_dir.y * along, z=0.03)
            tie_a = tie_center - carla.Location(x=road_fwd.x * 2.2, y=road_fwd.y * 2.2, z=0.0)
            tie_b = tie_center + carla.Location(x=road_fwd.x * 2.2, y=road_fwd.y * 2.2, z=0.0)
            debug.draw_line(tie_a, tie_b, thickness=0.05, color=tie_color, life_time=life_time)

        # Stop line marker
        sl_a = stopline_loc - carla.Location(x=right0.x * 2.2, y=right0.y * 2.2, z=0.0) + carla.Location(z=0.05)
        sl_b = stopline_loc + carla.Location(x=right0.x * 2.2, y=right0.y * 2.2, z=0.0) + carla.Location(z=0.05)
        debug.draw_line(sl_a, sl_b, thickness=0.18, color=carla.Color(255, 255, 255), life_time=life_time)

    # ---------- Warning lights as two "posts" using static vehicles with hazard lights ----------
    warning_posts = []
    warning_active = False
    warning_start_time = None

    def spawn_warning_post(side_sign: float):
        loc = tracks_center - carla.Location(
            x=forward0.x * LIGHT_POST_BEFORE_TRACKS_M,
            y=forward0.y * LIGHT_POST_BEFORE_TRACKS_M,
            z=0.0,
        )
        loc = loc + carla.Location(
            x=right0.x * LIGHT_POST_LATERAL_M * side_sign,
            y=right0.y * LIGHT_POST_LATERAL_M * side_sign,
            z=0.0,
        )
        loc.z += 0.2
        rot = carla.Rotation(yaw=tracks_heading_yaw, pitch=0.0, roll=0.0)
        tf = carla.Transform(loc, rot)

        bp = pick_vehicle_bp(bp_lib, ["vehicle.micro.microlino", "vehicle.citroen.c3", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_warning_post")
        except Exception:
            pass

        v = world.try_spawn_actor(bp, tf)
        if not v:
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

    # ---------- Gate arm: use a long vehicle as an arm, rotate it down over time ----------
    gate_vehicle = None
    gate_up_tf = None
    gate_down_tf = None
    gate_down = False

    def spawn_gate():
        nonlocal gate_vehicle, gate_up_tf, gate_down_tf

        pivot_loc = stopline_loc + carla.Location(
            x=right0.x * GATE_PIVOT_LATERAL_M,
            y=right0.y * GATE_PIVOT_LATERAL_M,
            z=0.0,
        )
        pivot_loc.z += 0.35

        # Use a long truck/van as the arm; keep physics off and animate yaw only.
        bp = pick_vehicle_bp(bp_lib, ["vehicle.carlamotors.carlacola", "vehicle.ford.ambulance", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_gate_arm")
        except Exception:
            pass

        # "Up" is rotated to align with road shoulder, "down" is across lane.
        yaw_across = tracks_heading_yaw + 90.0
        yaw_up = yaw_across - 75.0

        arm_center_down = pivot_loc + carla.Location(
            x=-right0.x * (GATE_ARM_LENGTH_M * 0.35),
            y=-right0.y * (GATE_ARM_LENGTH_M * 0.35),
            z=0.0,
        )

        arm_center_up = pivot_loc + carla.Location(
            x=-right0.x * (GATE_ARM_LENGTH_M * 0.65),
            y=-right0.y * (GATE_ARM_LENGTH_M * 0.65),
            z=0.8,
        )

        gate_down_tf = carla.Transform(arm_center_down, carla.Rotation(yaw=yaw_across, pitch=0.0, roll=0.0))
        gate_up_tf = carla.Transform(arm_center_up, carla.Rotation(yaw=yaw_up, pitch=0.0, roll=0.0))

        gate_vehicle = world.try_spawn_actor(bp, gate_up_tf)
        if not gate_vehicle:
            return
        actors_to_destroy.append(gate_vehicle)
        try:
            gate_vehicle.set_simulate_physics(False)
            gate_vehicle.set_autopilot(False)
        except Exception:
            pass
        set_vehicle_lights(gate_vehicle, int(carla.VehicleLightState.Position))

    spawn_gate()

    # ---------- Train-like actor crossing the road (reinforces urgency) ----------
    train = None
    train_active = False
    train_start_time = None

    def spawn_train_like_actor():
        nonlocal train
        # Move across the road through tracks_center using a long vehicle (bus/truck).
        bp = pick_vehicle_bp(bp_lib, ["vehicle.carlamotors.carlacola", "vehicle.bus.*", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_train_like")
        except Exception:
            pass

        # Direction across the road (use right vector as track direction)
        road_yaw = tracks_heading_yaw
        road_fwd = carla.Vector3D(x=math.cos(math.radians(road_yaw)), y=math.sin(math.radians(road_yaw)), z=0.0)
        road_right = carla.Vector3D(x=-road_fwd.y, y=road_fwd.x, z=0.0)
        track_dir = unit2(vec2(road_right))

        start_loc = tracks_center - carla.Location(x=track_dir.x * TRAIN_START_OFFSET_M, y=track_dir.y * TRAIN_START_OFFSET_M, z=0.0)
        start_loc.z += 0.3
        yaw = math.degrees(math.atan2(track_dir.y, track_dir.x))
        tf = carla.Transform(start_loc, carla.Rotation(yaw=yaw))

        train = world.try_spawn_actor(bp, tf)
        if not train:
            return
        actors_to_destroy.append(train)
        try:
            train.set_autopilot(False)
            train.set_simulate_physics(True)
        except Exception:
            pass

    spawn_train_like_actor()

    # ---------- Main loop (15 seconds) ----------
    start_wall = time.time()

    try:
        while True:
            wall_elapsed = time.time() - start_wall
            if wall_elapsed >= DURATION_SECONDS:
                break

            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds

            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            # Distance reference: stopline is where the decision point should occur
            dist_to_stopline = ego_loc.distance(stopline_loc)

            # Trigger warnings as ego enters decision zone
            if (warning_start_time is None) and (dist_to_stopline < DECISION_ZONE_START_M):
                warning_start_time = sim_time
                warning_active = True
                train_start_time = sim_time + 0.8  # start train shortly after warnings
                print("[EVENT] Railroad warning activated")

            # Flash warning lights using hazards (blinkers blink automatically if we set both)
            if warning_active and warning_posts:
                for post in warning_posts:
                    if not post:
                        continue
                    # Keep position lights on; add both blinkers to simulate alternating flash.
                    set_vehicle_lights(
                        post,
                        int(
                            carla.VehicleLightState.Position
                            | carla.VehicleLightState.LeftBlinker
                            | carla.VehicleLightState.RightBlinker
                        ),
                    )

            # Gate animation: rotate/teleport from up to down over time
            if gate_vehicle and warning_start_time is not None and not gate_down:
                t = (sim_time - warning_start_time) / max(GATE_FULLY_DOWN_AFTER, 1e-3)
                t = clamp(t, 0.0, 1.0)

                # Interpolate location and yaw
                loc = carla.Location(
                    x=gate_up_tf.location.x * (1 - t) + gate_down_tf.location.x * t,
                    y=gate_up_tf.location.y * (1 - t) + gate_down_tf.location.y * t,
                    z=gate_up_tf.location.z * (1 - t) + gate_down_tf.location.z * t,
                )
                yaw = gate_up_tf.rotation.yaw * (1 - t) + gate_down_tf.rotation.yaw * t
                tf = carla.Transform(loc, carla.Rotation(yaw=yaw))
                try:
                    gate_vehicle.set_transform(tf)
                except Exception:
                    pass

                if t >= 0.999:
                    gate_down = True
                    print("[EVENT] Gate down (lane blocked)")

            # Train motion across tracks
            if train and (train_start_time is not None) and (sim_time >= train_start_time):
                if not train_active:
                    train_active = True
                    print("[EVENT] Train approaching / crossing")

                if sim_time - train_start_time <= TRAIN_CROSSING_WINDOW:
                    road_yaw = tracks_heading_yaw
                    road_fwd = carla.Vector3D(x=math.cos(math.radians(road_yaw)), y=math.sin(math.radians(road_yaw)), z=0.0)
                    road_right = carla.Vector3D(x=-road_fwd.y, y=road_fwd.x, z=0.0)
                    track_dir = unit2(vec2(road_right))
                    try:
                        train.set_target_velocity(track_dir * TRAIN_SPEED_MPS)
                    except Exception:
                        pass
                else:
                    try:
                        train.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    except Exception:
                        pass

            # Camera: slightly above and behind ego, aligned with ego yaw (forward-facing)
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_loc - forward * 7.0 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-10.0, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # Optional visuals (if enabled)
            draw_crossing_visuals(life_time=0.11)

            # Tick & record
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
            try:
                if a:
                    a.destroy()
            except Exception:
                pass

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
