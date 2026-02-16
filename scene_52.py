import os
import time
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA, location_to_waypoint, route_maker
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================

# Networking
HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))
TM_PORT = int(os.getenv("CARLA_TM_PORT", 8000))

# Simulation
MAP_NAME = "Town04"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
DURATION_SEC = 15.0

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# Ego + route
EGO_SPAWN_INDEX = 0
ROUTE_END_INDEX = 20
ROUTE_XML_PATH = "/tmp/pcla_route_cut_in.xml"

# Cut-in vehicle behavior
CUTIN_SPAWN_AFTER_SEC = 2.0     # spawn after simulation starts
CUTIN_SAME_DIR_AHEAD = 12.0     # meters ahead of ego when spawned
CUTIN_RIGHT_OFFSET = 3.6        # meters to the right (approx one lane)
CUTIN_SPEED = 17.0              # m/s (~61 km/h)
CUTIN_START_LANECHANGE_AFTER = 2.0  # seconds after spawn to start moving left into ego lane
CUTIN_LANECHANGE_DURATION = 2.2     # seconds to complete lateral shift

# Camera follow
CAM_BACK = 7.0
CAM_UP = 3.3
CAM_PITCH = -12.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def safe_stop_destroy(sensor):
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    safe_destroy(sensor)


def choose_spawn_points(world: carla.World, ego_idx: int, end_idx: int):
    sp = world.get_map().get_spawn_points()
    if not sp:
        raise RuntimeError("No spawn points found on this map.")
    ego_idx = max(0, min(ego_idx, len(sp) - 1))
    end_idx = max(0, min(end_idx, len(sp) - 1))
    if end_idx == ego_idx:
        end_idx = (ego_idx + 10) % len(sp)
    return sp[ego_idx], sp[end_idx]


def make_route_xml(client: carla.Client, start_tf: carla.Transform, end_tf: carla.Transform, out_path: str):
    waypoints = location_to_waypoint(client, start_tf.location, end_tf.location)
    route_maker(waypoints, out_path)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    spectator = world.get_spectator()

    ego = None
    cutin = None
    camera = None
    video = None

    image_queue: Queue = Queue()

    try:
        ego_spawn, end_spawn = choose_spawn_points(world, EGO_SPAWN_INDEX, ROUTE_END_INDEX)
        make_route_xml(client, ego_spawn, end_spawn, ROUTE_XML_PATH)

        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)

        world.tick()

        pcla = PCLA("carl_carlv11", ego, ROUTE_XML_PATH, client)

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

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

        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        cutin_spawn_time = None
        lanechange_start_time = None

        while True:
            now = world.get_snapshot().timestamp.elapsed_seconds
            t = now - start_sim_time
            if t >= DURATION_SEC:
                break

            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_fwd = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()

            if cutin is None and t >= CUTIN_SPAWN_AFTER_SEC:
                spawn_loc = ego_loc + ego_fwd * CUTIN_SAME_DIR_AHEAD + ego_right * CUTIN_RIGHT_OFFSET
                spawn_loc.z += 0.3
                spawn_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_tf.rotation.yaw))

                cutin_bp_candidates = blueprints.filter("vehicle.*")
                preferred = [bp for bp in cutin_bp_candidates if "audi" in bp.id or "bmw" in bp.id or "toyota" in bp.id]
                cutin_bp = preferred[0] if preferred else blueprints.filter("vehicle.audi.*")[0]

                cutin = world.try_spawn_actor(cutin_bp, spawn_tf)
                if cutin is None:
                    cutin = world.spawn_actor(cutin_bp, spawn_tf)

                cutin.set_autopilot(False)
                cutin.set_simulate_physics(True)
                cutin_spawn_time = t
                lanechange_start_time = None

            if cutin is not None and cutin_spawn_time is not None:
                if lanechange_start_time is None and (t - cutin_spawn_time) >= CUTIN_START_LANECHANGE_AFTER:
                    lanechange_start_time = t

                cutin_tf = cutin.get_transform()
                cutin_loc = cutin_tf.location

                forward_vel = carla.Vector3D(ego_fwd.x * CUTIN_SPEED, ego_fwd.y * CUTIN_SPEED, 0.0)
                cutin.set_target_velocity(forward_vel)

                if lanechange_start_time is not None:
                    u = (t - lanechange_start_time) / max(1e-6, CUTIN_LANECHANGE_DURATION)
                    u = max(0.0, min(1.0, u))

                    lateral = CUTIN_RIGHT_OFFSET * (1.0 - u)
                    desired_loc = ego_loc + ego_fwd * CUTIN_SAME_DIR_AHEAD + ego_right * lateral
                    desired_loc.z = cutin_loc.z

                    desired_tf = carla.Transform(desired_loc, carla.Rotation(yaw=ego_tf.rotation.yaw))
                    cutin.set_transform(desired_tf)

            cam_loc = ego_loc - ego_fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            world.tick()

            while not image_queue.empty():
                frame = image_queue.get()
                video.write(frame)

    finally:
        try:
            if video is not None:
                video.release()
        except Exception:
            pass

        try:
            if "pcla" in locals():
                pcla.cleanup()
        except Exception:
            pass

        safe_stop_destroy(camera)
        safe_destroy(cutin)
        safe_destroy(ego)

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
