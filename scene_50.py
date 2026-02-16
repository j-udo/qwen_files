import os
import math
import random
import time
from queue import Queue

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA, location_to_waypoint, route_maker

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))
TM_PORT = int(os.getenv("CARLA_TM_PORT", 8000))
MAP_NAME = os.getenv("CARLA_MAP", "Town04")

FIXED_DELTA = 0.05
FPS = int(round(1.0 / FIXED_DELTA))
DURATION_SEC = 15.0

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

EGO_BP_FILTER = os.getenv("EGO_BP_FILTER", "vehicle.tesla.model3")
EGO_SPAWN_INDEX = int(os.getenv("EGO_SPAWN_INDEX", "0"))

ROUTE_FILE = "/tmp/pcla_route_cutin.xml"
ROUTE_DISTANCE_AHEAD = 250.0

CUTIN_BP_FILTER = os.getenv("CUTIN_BP_FILTER", "vehicle.audi.a2")
CUTIN_SPAWN_AFTER_SEC = 2.0
SPAWN_AHEAD_MIN = 18.0
SPAWN_AHEAD_MAX = 26.0

CUTIN_LANE_CHANGE_DELAY = 1.2
CUTIN_LANE_MERGE_TIME = 2.2
CUTIN_TARGET_GAP = 8.0
CUTIN_MIN_GAP = 5.0
CUTIN_SPEED_BOOST = 3.0
CUTIN_MAX_SPEED = 40.0

CAM_BACK = 7.0
CAM_UP = 3.2
CAM_PITCH = -12.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def speed_mps(actor):
    v = actor.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def safe_stop_and_destroy(sensor):
    try:
        if sensor is not None:
            sensor.stop()
            sensor.destroy()
    except Exception:
        pass


def setup_world(client):
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # ensure it renders
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def build_route_file(client, world, start_tf):
    m = world.get_map()
    start_loc = start_tf.location
    start_wp = m.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if start_wp is None:
        raise RuntimeError("Could not find a driving waypoint for ego spawn.")

    cur = start_wp
    remaining = ROUTE_DISTANCE_AHEAD
    step = 5.0
    while remaining > 0.0:
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        remaining -= step

    end_loc = cur.transform.location
    wps = location_to_waypoint(client, start_loc, end_loc)
    route_maker(wps, ROUTE_FILE)
    return ROUTE_FILE


def try_find_right_lane_spawn_transform(world, ego_tf):
    m = world.get_map()
    ego_wp = m.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return None

    right_wp = ego_wp.get_right_lane()
    if right_wp is None or right_wp.lane_type != carla.LaneType.Driving:
        return None

    dist_ahead = random.uniform(SPAWN_AHEAD_MIN, SPAWN_AHEAD_MAX)
    candidates = right_wp.next(dist_ahead)
    if not candidates:
        return None

    spawn_tf = candidates[0].transform
    spawn_tf.location.z += 0.35
    return spawn_tf


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = None
    ego = None
    cutin = None
    camera = None
    pcla = None
    video = None

    image_queue = Queue()

    cutin_spawn_time = None
    cutin_merge_start_time = None
    cutin_merge_done = False

    try:
        world = setup_world(client)
        blueprints = world.get_blueprint_library()

        spawns = world.get_map().get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points found in map.")

        ego_spawn_index = int(clamp(EGO_SPAWN_INDEX, 0, len(spawns) - 1))
        ego_spawn = spawns[ego_spawn_index]

        ego_bp_list = blueprints.filter(EGO_BP_FILTER)
        if not ego_bp_list:
            ego_bp_list = blueprints.filter("vehicle.tesla.model3")
        ego_bp = ego_bp_list[0]
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            for tf in spawns:
                ego = world.try_spawn_actor(ego_bp, tf)
                if ego is not None:
                    ego_spawn = tf
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        world.tick()

        route_file = build_route_file(client, world, ego_spawn)
        pcla = PCLA("carl_carlv11", ego, route_file, client)

        spectator = world.get_spectator()

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            frame = arr[:, :, :3]  # BG R? CARLA provides BGRA; keep as BGR for cv2
            image_queue.put(frame)

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds

        while True:
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds
            elapsed = sim_time - start_sim_time
            if elapsed >= DURATION_SEC:
                break

            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_forward = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()

            if cutin is None and elapsed >= CUTIN_SPAWN_AFTER_SEC:
                spawn_tf = try_find_right_lane_spawn_transform(world, ego_tf)
                if spawn_tf is not None:
                    cutin_bp_list = blueprints.filter(CUTIN_BP_FILTER)
                    if not cutin_bp_list:
                        cutin_bp_list = blueprints.filter("vehicle.audi.*")
                    cutin_bp = cutin_bp_list[0]
                    if cutin_bp.has_attribute("role_name"):
                        cutin_bp.set_attribute("role_name", "scenario")

                    cutin = world.try_spawn_actor(cutin_bp, spawn_tf)
                    if cutin is not None:
                        cutin.set_autopilot(False)
                        cutin.set_simulate_physics(True)
                        cutin_spawn_time = sim_time
                        cutin_merge_start_time = None
                        cutin_merge_done = False

            if cutin is not None:
                cutin_tf = cutin.get_transform()
                cutin_loc = cutin_tf.location

                cutin_wp = world.get_map().get_waypoint(
                    cutin_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                lane_forward = cutin_tf.get_forward_vector()
                if cutin_wp is not None:
                    lane_forward = cutin_wp.transform.get_forward_vector()

                ego_spd = speed_mps(ego)
                desired_speed = min(CUTIN_MAX_SPEED, ego_spd + CUTIN_SPEED_BOOST)

                rel = cutin_loc - ego_loc
                forward_gap = rel.x * ego_forward.x + rel.y * ego_forward.y + rel.z * ego_forward.z

                if forward_gap < CUTIN_TARGET_GAP:
                    desired_speed = min(desired_speed, max(ego_spd, 0.0))

                if cutin_merge_start_time is None and cutin_spawn_time is not None:
                    if (sim_time - cutin_spawn_time) >= CUTIN_LANE_CHANGE_DELAY:
                        cutin_merge_start_time = sim_time

                lateral_vec = carla.Vector3D(0.0, 0.0, 0.0)
                if cutin_merge_start_time is not None and not cutin_merge_done:
                    t = (sim_time - cutin_merge_start_time) / max(CUTIN_LANE_MERGE_TIME, 0.001)
                    t = clamp(t, 0.0, 1.0)

                    smooth = t * t * (3.0 - 2.0 * t)  # smoothstep

                    # From right lane to ego lane => move left in world, which is -ego_right
                    lateral_dir = carla.Vector3D(-ego_right.x, -ego_right.y, 0.0)

                    lateral_speed = 2.8 * (1.0 - (smooth - 0.5) * (smooth - 0.5) * 4.0)
                    lateral_speed = clamp(lateral_speed, 0.0, 3.0)
                    lateral_vec = lateral_dir * lateral_speed

                    if t >= 1.0:
                        cutin_merge_done = True

                if forward_gap < CUTIN_MIN_GAP:
                    lateral_vec = carla.Vector3D(0.0, 0.0, 0.0)
                    desired_speed = min(desired_speed, max(ego_spd, 0.0))

                forward_vec = carla.Vector3D(lane_forward.x, lane_forward.y, 0.0) * float(desired_speed)
                cutin.set_target_velocity(forward_vec + lateral_vec)

            cam_loc = ego_loc - ego_forward * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            world.tick()

            # record any queued frames for this tick
            while not image_queue.empty():
                video.write(image_queue.get())

        # give sensors a couple ticks to flush frames
        for _ in range(2):
            world.tick()
            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        safe_stop_and_destroy(camera)

        try:
            if video is not None:
                video.release()
        except Exception:
            pass

        try:
            if pcla is not None:
                pcla.cleanup()
        except Exception:
            pass

        safe_destroy(cutin)
        safe_destroy(ego)

        try:
            tm = client.get_trafficmanager(TM_PORT)
            tm.set_synchronous_mode(False)
        except Exception:
            pass

        try:
            if world is not None:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.no_rendering_mode = False
                world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()