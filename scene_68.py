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

MAP_NAME = os.getenv("CARLA_MAP", "Town03")

FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

DURATION_SECONDS = 15.0

# Scenario geometry/logic (relative to ego at start)
CROSSING_AHEAD_M = 70.0
STOPLINE_BEFORE_TRACKS_M = 14.0

# When to begin warning activation relative to stopline
DECISION_ZONE_START_M = 55.0

# Warning system timings
LIGHTS_START_IMMEDIATELY = True
GATE_FULLY_DOWN_AFTER_S = 3.0
TRAIN_START_DELAY_AFTER_WARNING_S = 1.0
TRAIN_CROSSING_WINDOW_S = 7.0

# Visual proportions (debug-draw)
TRACKS_HALF_WIDTH_M = 9.0
TRACK_GAUGE_OFFSET_M = 0.85
TIE_HALF_LENGTH_M = 2.8
TIE_SPACING_M = 0.75
TIES_COUNT_EACH_SIDE = 12

# Place warning posts and gate near stopline
LIGHT_POST_BEFORE_TRACKS_M = 8.0
LIGHT_POST_LATERAL_M = 6.0
GATE_PIVOT_LATERAL_M = 4.0
GATE_ARM_LENGTH_M = 7.5

# Train-like actor
TRAIN_START_OFFSET_M = 60.0
TRAIN_SPEED_MPS = 14.0

DEBUG_DRAW = True
DEBUG_LIFETIME = 0.20


def vec2(v: carla.Vector3D) -> carla.Vector3D:
    return carla.Vector3D(v.x, v.y, 0.0)


def unit2(v: carla.Vector3D) -> carla.Vector3D:
    n = math.hypot(v.x, v.y)
    if n < 1e-6:
        return carla.Vector3D(1.0, 0.0, 0.0)
    return carla.Vector3D(v.x / n, v.y / n, 0.0)


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def pick_bp(bp_lib: carla.BlueprintLibrary, patterns):
    for p in patterns:
        bps = bp_lib.filter(p)
        if bps:
            return bps[0]
    all_bps = bp_lib.filter("*")
    return all_bps[0]


def set_vehicle_lights(vehicle: carla.Vehicle, mask: int) -> None:
    try:
        vehicle.set_light_state(carla.VehicleLightState(mask))
    except Exception:
        pass


def setup_world(client: carla.Client) -> carla.World:
    world = client.get_world()
    current = ""
    try:
        current = world.get_map().name
    except Exception:
        pass

    if MAP_NAME not in current:
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


def find_good_spawn(world: carla.World, preferred_index: int = 31) -> carla.Transform:
    m = world.get_map()
    sps = m.get_spawn_points()
    if not sps:
        raise RuntimeError("No spawn points found")

    if 0 <= preferred_index < len(sps):
        return sps[preferred_index]

    return sps[0]


def compute_crossing_reference(world: carla.World, ego_tf: carla.Transform):
    m = world.get_map()
    forward0 = unit2(vec2(ego_tf.get_forward_vector()))
    right0 = unit2(vec2(ego_tf.get_right_vector()))

    approx_tracks_center = ego_tf.location + carla.Location(
        x=forward0.x * CROSSING_AHEAD_M,
        y=forward0.y * CROSSING_AHEAD_M,
        z=0.0,
    )

    tracks_wp = m.get_waypoint(
        approx_tracks_center,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if tracks_wp is None:
        tracks_center = approx_tracks_center
        road_yaw = ego_tf.rotation.yaw
        road_fwd = forward0
        road_right = right0
    else:
        tracks_center = tracks_wp.transform.location
        road_yaw = tracks_wp.transform.rotation.yaw
        road_fwd = unit2(vec2(tracks_wp.transform.get_forward_vector()))
        road_right = unit2(vec2(tracks_wp.transform.get_right_vector()))

    track_dir = road_right

    stopline_loc = tracks_center - carla.Location(
        x=road_fwd.x * STOPLINE_BEFORE_TRACKS_M,
        y=road_fwd.y * STOPLINE_BEFORE_TRACKS_M,
        z=0.0,
    )
    stop_wp = m.get_waypoint(stopline_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if stop_wp is not None:
        stopline_loc = stop_wp.transform.location

    return tracks_center, stopline_loc, road_fwd, road_right, road_yaw, track_dir


def draw_crossing(debug: carla.DebugHelper, tracks_center, stopline_loc, road_fwd, road_right, track_dir, warning_active: bool):
    if not DEBUG_DRAW:
        return

    z = 0.08
    rail_color = carla.Color(210, 210, 210)
    tie_color = carla.Color(110, 70, 40)
    stop_color = carla.Color(245, 245, 245)
    warn_color = carla.Color(255, 40, 40)
    text_color = carla.Color(250, 250, 250)

    # Rails: 2 thin bright lines across the road
    for offset in (-TRACK_GAUGE_OFFSET_M, TRACK_GAUGE_OFFSET_M):
        c = tracks_center + carla.Location(z=z) + carla.Location(
            x=road_fwd.x * offset,
            y=road_fwd.y * offset,
            z=0.0,
        )
        p0 = c - carla.Location(x=track_dir.x * TRACKS_HALF_WIDTH_M, y=track_dir.y * TRACKS_HALF_WIDTH_M, z=0.0)
        p1 = c + carla.Location(x=track_dir.x * TRACKS_HALF_WIDTH_M, y=track_dir.y * TRACKS_HALF_WIDTH_M, z=0.0)
        debug.draw_line(p0, p1, thickness=0.14, color=rail_color, life_time=DEBUG_LIFETIME)

    # Ties
    for k in range(-TIES_COUNT_EACH_SIDE, TIES_COUNT_EACH_SIDE + 1):
        along = k * TIE_SPACING_M
        tie_center = tracks_center + carla.Location(x=track_dir.x * along, y=track_dir.y * along, z=z - 0.02)
        a = tie_center - carla.Location(x=road_fwd.x * TIE_HALF_LENGTH_M, y=road_fwd.y * TIE_HALF_LENGTH_M, z=0.0)
        b = tie_center + carla.Location(x=road_fwd.x * TIE_HALF_LENGTH_M, y=road_fwd.y * TIE_HALF_LENGTH_M, z=0.0)
        debug.draw_line(a, b, thickness=0.06, color=tie_color, life_time=DEBUG_LIFETIME)

    # Stop line
    sl_a = stopline_loc - carla.Location(x=road_right.x * 2.6, y=road_right.y * 2.6, z=0.0) + carla.Location(z=z)
    sl_b = stopline_loc + carla.Location(x=road_right.x * 2.6, y=road_right.y * 2.6, z=0.0) + carla.Location(z=z)
    debug.draw_line(sl_a, sl_b, thickness=0.22, color=stop_color, life_time=DEBUG_LIFETIME)

    # Labels
    debug.draw_string(stopline_loc + carla.Location(z=1.7), "STOP LINE", draw_shadow=True, color=text_color, life_time=DEBUG_LIFETIME)
    debug.draw_string(tracks_center + carla.Location(z=1.7), "RAIL TRACKS", draw_shadow=True, color=text_color, life_time=DEBUG_LIFETIME)

    if warning_active:
        debug.draw_string(tracks_center + carla.Location(z=2.4), "WARNING: TRAIN", draw_shadow=True, color=warn_color, life_time=DEBUG_LIFETIME)


def spawn_warning_post(world: carla.World, bp_lib: carla.BlueprintLibrary, base_loc: carla.Location, road_yaw: float):
    # Use a small vehicle as a visible "signal cabinet"; blinkers simulate flashing lights
    bp = pick_bp(bp_lib, ["vehicle.micro.microlino", "vehicle.citroen.c3", "vehicle.*"])
    try:
        bp.set_attribute("role_name", "rr_warning_post")
    except Exception:
        pass

    tf = carla.Transform(base_loc, carla.Rotation(yaw=road_yaw))
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        return None

    try:
        actor.set_simulate_physics(False)
        actor.set_autopilot(False)
    except Exception:
        pass

    set_vehicle_lights(actor, int(carla.VehicleLightState.Position))
    return actor


def spawn_gate_arm(world: carla.World, bp_lib: carla.BlueprintLibrary, up_tf: carla.Transform):
    # Use a long vehicle as a visible barrier arm approximation (static; teleported for animation)
    bp = pick_bp(bp_lib, ["vehicle.carlamotors.carlacola", "vehicle.ford.ambulance", "vehicle.*"])
    try:
        bp.set_attribute("role_name", "rr_gate_arm")
    except Exception:
        pass

    gate = world.try_spawn_actor(bp, up_tf)
    if gate is None:
        return None

    try:
        gate.set_simulate_physics(False)
        gate.set_autopilot(False)
    except Exception:
        pass

    set_vehicle_lights(gate, int(carla.VehicleLightState.Position))
    return gate


def spawn_train_like(world: carla.World, bp_lib: carla.BlueprintLibrary, tf: carla.Transform):
    bp = pick_bp(bp_lib, ["vehicle.bus.*", "vehicle.carlamotors.carlacola", "vehicle.*"])
    try:
        bp.set_attribute("role_name", "rr_train_like")
    except Exception:
        pass

    train = world.try_spawn_actor(bp, tf)
    if train is None:
        return None

    try:
        train.set_autopilot(False)
        train.set_simulate_physics(True)
    except Exception:
        pass

    set_vehicle_lights(train, int(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
    return train


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = setup_world(client)
    bp_lib = world.get_blueprint_library()
    debug = world.debug
    m = world.get_map()

    actors_to_destroy = []

    # Spawn ego
    ego_bp = pick_bp(bp_lib, ["vehicle.tesla.model3", "vehicle.*"])
    try:
        ego_bp.set_attribute("role_name", "hero")
    except Exception:
        pass

    ego_spawn = find_good_spawn(world, preferred_index=31)
    ego = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego is None:
        ego = world.spawn_actor(ego_bp, ego_spawn)
    actors_to_destroy.append(ego)
    world.tick()

    # PCLA agent
    route_xml = "./sample_route.xml"
    pcla = PCLA("carl_carlv11", ego, route_xml, client)

    # Encourage moderate speed while keeping rule following
    try:
        tm = client.get_trafficmanager(TM_PORT)
        tm.vehicle_percentage_speed_difference(ego, -5.0)
        tm.ignore_lights_percentage(ego, 0.0)
        tm.ignore_signs_percentage(ego, 0.0)
        tm.auto_lane_change(ego, True)
        tm.update_vehicle_lights(ego, False)
    except Exception:
        tm = None

    # Camera: attach to ego (stable rear chase view, aligned with ego perspective)
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    cam_rel_tf = carla.Transform(carla.Location(x=-7.0, y=0.0, z=3.2), carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0))
    camera = world.spawn_actor(cam_bp, cam_rel_tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)
    actors_to_destroy.append(camera)

    image_queue: Queue = Queue()

    def camera_callback(image: carla.Image) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        image_queue.put(arr[:, :, :3])

    camera.listen(camera_callback)

    video = cv2.VideoWriter(
        VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (IMAGE_W, IMAGE_H),
    )

    # Compute crossing reference off ego start
    ego_start_tf = ego.get_transform()
    tracks_center, stopline_loc, road_fwd, road_right, road_yaw, track_dir = compute_crossing_reference(world, ego_start_tf)

    # Warning posts (left/right)
    warning_posts = []

    for side in (1.0, -1.0):
        base = tracks_center - carla.Location(x=road_fwd.x * LIGHT_POST_BEFORE_TRACKS_M, y=road_fwd.y * LIGHT_POST_BEFORE_TRACKS_M, z=0.0)
        base = base + carla.Location(x=road_right.x * LIGHT_POST_LATERAL_M * side, y=road_right.y * LIGHT_POST_LATERAL_M * side, z=0.0)
        base.z += 0.25
        post = spawn_warning_post(world, bp_lib, base, road_yaw)
        if post is not None:
            warning_posts.append(post)
            actors_to_destroy.append(post)

    # Gate arm (up and down transforms)
    gate = None
    gate_up_tf = None
    gate_down_tf = None
    gate_down = False

    pivot_loc = stopline_loc + carla.Location(x=road_right.x * GATE_PIVOT_LATERAL_M, y=road_right.y * GATE_PIVOT_LATERAL_M, z=0.0)
    pivot_loc.z += 0.45

    # down: across road, up: along shoulder (angled away)
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

    gate = spawn_gate_arm(world, bp_lib, gate_up_tf)
    if gate is not None:
        actors_to_destroy.append(gate)

    # Train-like actor
    train = None
    train_active = False
    train_start_time = None

    train_start_loc = tracks_center - carla.Location(x=track_dir.x * TRAIN_START_OFFSET_M, y=track_dir.y * TRAIN_START_OFFSET_M, z=0.0)
    train_start_loc.z += 0.35
    train_yaw = math.degrees(math.atan2(track_dir.y, track_dir.x))
    train_tf = carla.Transform(train_start_loc, carla.Rotation(yaw=train_yaw))
    train = spawn_train_like(world, bp_lib, train_tf)
    if train is not None:
        actors_to_destroy.append(train)

    # Warning logic
    warning_active = False
    warning_start_time = None

    start_wall = time.time()

    try:
        while True:
            wall_elapsed = time.time() - start_wall
            if wall_elapsed >= DURATION_SECONDS:
                break

            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds

            # Ego control (PCLA)
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_loc = ego.get_location()
            dist_to_stopline = ego_loc.distance(stopline_loc)

            # Activate warning as ego enters decision zone
            if warning_start_time is None and dist_to_stopline < DECISION_ZONE_START_M:
                warning_start_time = sim_time
                warning_active = True
                train_start_time = sim_time + TRAIN_START_DELAY_AFTER_WARNING_S

            # Flashing lights (hazards)
            if warning_active and LIGHTS_START_IMMEDIATELY:
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

            # Gate animation
            if gate is not None and warning_start_time is not None and not gate_down:
                t = (sim_time - warning_start_time) / max(GATE_FULLY_DOWN_AFTER_S, 1e-3)
                t = clamp(t, 0.0, 1.0)

                loc = carla.Location(
                    x=gate_up_tf.location.x * (1 - t) + gate_down_tf.location.x * t,
                    y=gate_up_tf.location.y * (1 - t) + gate_down_tf.location.y * t,
                    z=gate_up_tf.location.z * (1 - t) + gate_down_tf.location.z * t,
                )
                yaw = gate_up_tf.rotation.yaw * (1 - t) + gate_down_tf.rotation.yaw * t
                tf = carla.Transform(loc, carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0))
                try:
                    gate.set_transform(tf)
                except Exception:
                    pass

                if t >= 0.999:
                    gate_down = True

            # Train motion across the road at the tracks
            if train is not None and train_start_time is not None and sim_time >= train_start_time:
                if not train_active:
                    train_active = True

                if sim_time - train_start_time <= TRAIN_CROSSING_WINDOW_S:
                    try:
                        train.set_target_velocity(track_dir * TRAIN_SPEED_MPS)
                    except Exception:
                        pass
                else:
                    try:
                        train.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    except Exception:
                        pass

            # Draw visible crossing cues
            draw_crossing(debug, tracks_center, stopline_loc, road_fwd, road_right, track_dir, warning_active)

            # Keep spectator behind ego too (so the CARLA window view matches the recorded view closely)
            spectator = world.get_spectator()
            ego_tf = ego.get_transform()
            fwd = ego_tf.get_forward_vector()
            cam_loc = ego_tf.location - fwd * 7.0 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-12.0, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # Tick and record
            world.tick()
            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        try:
            if camera is not None:
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
