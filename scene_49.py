import os
import time
from queue import Queue, Empty
from typing import Optional

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", "2000"))

MAP_NAME = "Town03"
ROUTE_XML = "./sample_route.xml"
PCLA_AGENT = "carl_carlv11"

FIXED_DELTA = 0.05
FPS = int(round(1.0 / FIXED_DELTA))

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

TERMINATE_AFTER_SECONDS = 15.0

ONCOMING_SPAWN_AFTER_SECONDS = 1.0
ONCOMING_DISTANCE_AHEAD = 60.0
ONCOMING_SPEED = 16.0  # m/s

CAM_BACK = 7.5
CAM_UP = 3.2
CAM_PITCH = -12.0

EGO_SPAWN_INDEX = 31


def safe_stop(actor):
    try:
        if actor is not None:
            actor.stop()
    except Exception:
        pass


def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def setup_world(client: carla.Client) -> carla.World:
    print("[INFO] Loading map:", MAP_NAME)
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # ensure viewers see rendering
    world.apply_settings(settings)

    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world


def pick_ego_spawn(carla_map: carla.Map) -> carla.Transform:
    sp = carla_map.get_spawn_points()
    if not sp:
        raise RuntimeError("No spawn points found in map.")
    idx = EGO_SPAWN_INDEX if len(sp) > EGO_SPAWN_INDEX else 0
    return sp[idx]


def spawn_ego(world: carla.World, blueprints: carla.BlueprintLibrary, carla_map: carla.Map) -> carla.Vehicle:
    ego_candidates = blueprints.filter("vehicle.tesla.model3")
    if not ego_candidates:
        ego_candidates = blueprints.filter("vehicle.*")
    ego_bp = ego_candidates[0]
    try:
        ego_bp.set_attribute("role_name", "hero")
    except Exception:
        pass

    ego = world.try_spawn_actor(ego_bp, pick_ego_spawn(carla_map))
    if ego is None:
        for tf in carla_map.get_spawn_points():
            ego = world.try_spawn_actor(ego_bp, tf)
            if ego is not None:
                break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    ego.set_autopilot(False)
    ego.set_simulate_physics(True)
    world.tick()
    return ego


def chase_cam_transform(ego_tf: carla.Transform) -> carla.Transform:
    forward = ego_tf.get_forward_vector()
    cam_loc = ego_tf.location - forward * CAM_BACK + carla.Location(z=CAM_UP)
    cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
    return carla.Transform(cam_loc, cam_rot)


def setup_rgb_camera(world: carla.World, blueprints: carla.BlueprintLibrary, image_queue: Queue) -> carla.Sensor:
    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(cam_bp, carla.Transform())

    def _cb(image: carla.Image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        bgr = arr[:, :, :3]  # sensor provides BGRA, OpenCV expects BGR
        image_queue.put(bgr)

    camera.listen(_cb)
    world.tick()
    return camera


def _find_oncoming_spawn_transform(
    carla_map: carla.Map, ego_loc: carla.Location, distance_ahead: float
) -> Optional[carla.Transform]:
    ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        return None

    next_wps = ego_wp.next(distance_ahead)
    if not next_wps:
        return None

    # Start from a waypoint ahead on ego's current driving lane.
    ahead_wp = next_wps[0]
    spawn_tf = ahead_wp.transform
    spawn_tf.location.z += 0.35

    # Make the other car face toward ego (wrong-way on the same lane centerline).
    spawn_tf.rotation.yaw = (spawn_tf.rotation.yaw + 180.0) % 360.0
    return spawn_tf


def spawn_oncoming(
    world: carla.World, blueprints: carla.BlueprintLibrary, carla_map: carla.Map, ego: carla.Vehicle
) -> Optional[carla.Vehicle]:
    spawn_tf = _find_oncoming_spawn_transform(carla_map, ego.get_location(), ONCOMING_DISTANCE_AHEAD)
    if spawn_tf is None:
        print("[WARN] Could not find oncoming spawn transform.")
        return None

    candidates = blueprints.filter("vehicle.audi.*")
    if not candidates:
        candidates = blueprints.filter("vehicle.*")
    bp = candidates[0]
    try:
        bp.set_attribute("role_name", "oncoming")
    except Exception:
        pass

    oncoming = world.try_spawn_actor(bp, spawn_tf)
    if oncoming is None:
        print("[WARN] Failed to spawn oncoming vehicle.")
        return None

    oncoming.set_autopilot(False)
    oncoming.set_simulate_physics(True)
    world.tick()

    print("[EVENT] Oncoming vehicle spawned head-on (wrong-way in ego lane)")
    return oncoming


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = None
    ego = None
    oncoming = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    try:
        world = setup_world(client)
        carla_map = world.get_map()
        blueprints = world.get_blueprint_library()
        spectator = world.get_spectator()

        ego = spawn_ego(world, blueprints, carla_map)

        pcla = PCLA(PCLA_AGENT, ego, ROUTE_XML, client)
        print("[INFO] Ego spawned, PCLA running")

        camera = setup_rgb_camera(world, blueprints, image_queue)

        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )
        if not video.isOpened():
            raise RuntimeError("Failed to open VideoWriter: %s" % VIDEO_PATH)

        # Prime spectator + camera for stable initial view
        for _ in range(10):
            ego_tf = ego.get_transform()
            cam_tf = chase_cam_transform(ego_tf)
            spectator.set_transform(cam_tf)
            camera.set_transform(cam_tf)
            world.tick()
            try:
                while True:
                    image_queue.get_nowait()
            except Empty:
                pass

        start_sim_t = world.get_snapshot().timestamp.elapsed_seconds
        oncoming_spawned = False

        while True:
            sim_t = world.get_snapshot().timestamp.elapsed_seconds
            elapsed = sim_t - start_sim_t
            if elapsed >= TERMINATE_AFTER_SECONDS:
                print("[INFO] Time limit reached, terminating scenario")
                break

            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            if (not oncoming_spawned) and elapsed >= ONCOMING_SPAWN_AFTER_SECONDS:
                oncoming = spawn_oncoming(world, blueprints, carla_map, ego)
                oncoming_spawned = oncoming is not None

            if oncoming is not None:
                fwd = oncoming.get_transform().get_forward_vector()
                oncoming.set_target_velocity(carla.Vector3D(fwd.x * ONCOMING_SPEED, fwd.y * ONCOMING_SPEED, 0.0))

            ego_tf = ego.get_transform()
            cam_tf = chase_cam_transform(ego_tf)
            spectator.set_transform(cam_tf)
            camera.set_transform(cam_tf)

            world.tick()

            # Record one frame per tick (best-effort)
            try:
                frame = image_queue.get(timeout=0.5)
                video.write(frame)
            except Empty:
                pass

    finally:
        print("[INFO] Cleaning up")

        safe_stop(camera)
        safe_destroy(camera)

        try:
            if video is not None:
                video.release()
                print("[INFO] Video saved:", VIDEO_PATH)
        except Exception:
            pass

        try:
            if pcla is not None:
                pcla.cleanup()
        except Exception:
            pass

        safe_destroy(oncoming)
        safe_destroy(ego)

        try:
            if world is not None:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.no_rendering_mode = False
                world.apply_settings(settings)
                try:
                    tm = client.get_trafficmanager()
                    tm.set_synchronous_mode(False)
                except Exception:
                    pass
        except Exception:
            pass


if __name__ == "__main__":
    main()