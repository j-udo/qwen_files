import os
import time
from queue import Queue, Empty

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", "2000"))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(round(1.0 / FIXED_DELTA))

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

TERMINATE_AFTER_SECONDS = 15.0

ROUTE_XML = "./sample_route.xml"
PCLA_AGENT = "carl_carlv11"

SPAWN_AFTER_SECONDS = 2.0
ONCOMING_DISTANCE_AHEAD = 70.0
ONCOMING_SPEED = 14.0  # m/s
MIN_GAP_TO_SPAWN = 25.0


def safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def safe_stop(sensor):
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass


def setup_world(client):
    client.get_world()

    print("[INFO] Loading map:", MAP_NAME)
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # ensure rendered for viewers
    world.apply_settings(settings)

    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world


def pick_ego_spawn(carla_map):
    sp = carla_map.get_spawn_points()
    if not sp:
        raise RuntimeError("No spawn points found in map.")
    idx = 31 if len(sp) > 31 else 0
    return sp[idx]


def spawn_ego(world, blueprints, carla_map):
    ego_bps = blueprints.filter("vehicle.tesla.model3")
    if not ego_bps:
        ego_bps = blueprints.filter("vehicle.*")
    ego_bp = ego_bps[0]
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
        raise RuntimeError("Failed to spawn ego vehicle at any spawn point.")

    ego.set_autopilot(False)
    ego.set_simulate_physics(True)
    world.tick()
    return ego


def chase_cam_transform(ego_tf):
    forward = ego_tf.get_forward_vector()
    loc = ego_tf.location - forward * 7.5 + carla.Location(z=3.2)
    rot = carla.Rotation(pitch=-12.0, yaw=ego_tf.rotation.yaw, roll=0.0)
    return carla.Transform(loc, rot)


def setup_rgb_camera(world, blueprints, image_queue):
    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(cam_bp, carla.Transform())

    def _cb(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3]
        image_queue.put(rgb)

    camera.listen(_cb)
    world.tick()
    return camera


def try_spawn_oncoming(world, carla_map, blueprints, ego_loc):
    ego_wp = carla_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        return None

    spawn_wp = None
    for dist in (ONCOMING_DISTANCE_AHEAD, ONCOMING_DISTANCE_AHEAD + 30.0, ONCOMING_DISTANCE_AHEAD + 60.0):
        nxt = ego_wp.next(dist)
        if not nxt:
            continue
        cand = nxt[0]
        if cand.transform.location.distance(ego_loc) >= MIN_GAP_TO_SPAWN:
            spawn_wp = cand
            break

    if spawn_wp is None:
        return None

    spawn_tf = spawn_wp.transform
    spawn_loc = carla.Location(spawn_tf.location.x, spawn_tf.location.y, spawn_tf.location.z + 0.35)
    spawn_rot = carla.Rotation(
        pitch=spawn_tf.rotation.pitch,
        yaw=spawn_tf.rotation.yaw + 180.0,
        roll=spawn_tf.rotation.roll,
    )

    oncoming_bps = blueprints.filter("vehicle.audi.*")
    if not oncoming_bps:
        oncoming_bps = blueprints.filter("vehicle.*")
    oncoming_bp = oncoming_bps[0]
    try:
        oncoming_bp.set_attribute("role_name", "oncoming")
    except Exception:
        pass

    oncoming = world.try_spawn_actor(oncoming_bp, carla.Transform(spawn_loc, spawn_rot))
    if oncoming is None:
        return None

    oncoming.set_autopilot(False)
    oncoming.set_simulate_physics(True)
    world.tick()
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

    image_queue = Queue()

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
            raise RuntimeError("Failed to open VideoWriter for %s" % VIDEO_PATH)

        for _ in range(5):
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
            snapshot = world.get_snapshot()
            sim_t = snapshot.timestamp.elapsed_seconds
            elapsed = sim_t - start_sim_t
            if elapsed >= TERMINATE_AFTER_SECONDS:
                print("[INFO] Time limit reached, terminating scenario")
                break

            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            if (not oncoming_spawned) and elapsed >= SPAWN_AFTER_SECONDS:
                oncoming = try_spawn_oncoming(world, carla_map, blueprints, ego_loc)
                if oncoming is not None:
                    oncoming_spawned = True
                    print("[EVENT] Oncoming vehicle spawned head-on")

            if oncoming is not None:
                fwd = oncoming.get_transform().get_forward_vector()
                oncoming.set_target_velocity(carla.Vector3D(fwd.x * ONCOMING_SPEED, fwd.y * ONCOMING_SPEED, 0.0))

            cam_tf = chase_cam_transform(ego_tf)
            spectator.set_transform(cam_tf)
            camera.set_transform(cam_tf)

            world.tick()

            try:
                frame = image_queue.get(timeout=0.2)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)
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
