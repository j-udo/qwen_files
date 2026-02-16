import os
import time
import random
import math
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA, route_maker, location_to_waypoint
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================

# Networking
HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

# Simulation
MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
DURATION_SECONDS = 15.0

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# Ego / route
EGO_SPAWN_INDEX = 31
EGO_END_INDEX = 42
ROUTE_PATH = "/tmp/pcla_route.xml"

# Jaywalking pedestrian behavior
JAYWALK_SPAWN_AHEAD_MIN = 18.0
JAYWALK_SPAWN_AHEAD_MAX = 26.0
JAYWALK_TRIGGER_DISTANCE_AHEAD = 22.0  # if spawn fails, keep trying until ego within this ahead distance to a good spot
CROSSING_OFFSET_RIGHT = 3.2            # meters from lane center towards sidewalk/right
CROSSING_DISTANCE_ACROSS = 10.0        # total distance to cross (right->left)
WALK_SPEED = 1.7                       # m/s

# Camera chase
CAM_BACK = 7.0
CAM_UP = 3.5
CAM_PITCH = -12.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    world.tick()
    return world


def try_spawn_pedestrian_crossing(world: carla.World, blueprints: carla.BlueprintLibrary, ego: carla.Vehicle) -> tuple:
    m = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    # pick a point ahead of ego on/near its lane
    ahead = random.uniform(JAYWALK_SPAWN_AHEAD_MIN, JAYWALK_SPAWN_AHEAD_MAX)
    target_center_loc = ego_loc + fwd * ahead

    wp_drive = m.get_waypoint(target_center_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp_drive is None:
        return None, None, None, None

    lane_center = wp_drive.transform.location

    # start from right side (near sidewalk/shoulder), then cross to left
    start_loc = lane_center + right * CROSSING_OFFSET_RIGHT
    end_loc = lane_center - right * (CROSSING_DISTANCE_ACROSS - CROSSING_OFFSET_RIGHT)

    # project to ground
    start_loc.z += 0.5
    end_loc.z += 0.5

    walker_bps = blueprints.filter("walker.pedestrian.*")
    if not walker_bps:
        return None, None, None, None

    walker_bp = random.choice(walker_bps)
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "true")

    walker_tf = carla.Transform(start_loc, carla.Rotation(yaw=ego_tf.rotation.yaw + 90.0))
    walker = world.try_spawn_actor(walker_bp, walker_tf)
    if walker is None:
        return None, None, None, None

    controller_bp = blueprints.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

    return walker, controller, start_loc, end_loc


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    spectator = world.get_spectator()

    ego = None
    pcla = None
    camera = None
    video = None

    walker = None
    walker_controller = None
    walker_end_loc = None
    walker_started = False

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        ego_spawn = spawn_points[min(EGO_SPAWN_INDEX, len(spawn_points) - 1)]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # ---------- Route for PCLA ----------
        start_loc = ego_spawn.location
        end_loc = spawn_points[min(EGO_END_INDEX, len(spawn_points) - 1)].location
        waypoints = location_to_waypoint(client, start_loc, end_loc)
        route_maker(waypoints, ROUTE_PATH)

        # ---------- PCLA agent ----------
        pcla = PCLA("carl_carlv11", ego, ROUTE_PATH, client)

        # ---------- Spectator Camera Sensor (for recording) ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

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

        sim_start = world.get_snapshot().timestamp.elapsed_seconds

        # allow jaywalking (must be set before pedestrians are spawned to affect general behavior; harmless here)
        world.set_pedestrians_cross_factor(1.0)

        last_spawn_attempt_time = -1.0

        while True:
            now = world.get_snapshot().timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # ===== Ego control via PCLA =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            fwd = ego_tf.get_forward_vector()

            # ===== Spawn jaywalker when ego is moving forward a bit =====
            # We attempt periodically until success.
            if walker is None:
                if last_spawn_attempt_time < 0 or (now - last_spawn_attempt_time) > 0.5:
                    last_spawn_attempt_time = now

                    # Try to spawn ahead along ego's current forward direction and near its lane
                    w, wc, w_start, w_end = try_spawn_pedestrian_crossing(world, blueprints, ego)
                    if w is not None:
                        walker = w
                        walker_controller = wc
                        walker_end_loc = w_end
                        walker_started = False

            # ===== Start pedestrian crossing when ego is close enough =====
            if walker is not None and (not walker_started):
                # Start when the crossing point is reasonably ahead of ego (so it's a "jaywalk as ego approaches")
                # Use the midpoint between start/end as the crossing center.
                cross_mid = carla.Location(
                    x=(walker.get_location().x + walker_end_loc.x) * 0.5,
                    y=(walker.get_location().y + walker_end_loc.y) * 0.5,
                    z=(walker.get_location().z + walker_end_loc.z) * 0.5,
                )

                # Compute signed forward distance (approx) using dot with ego forward
                to_mid = carla.Vector3D(
                    x=cross_mid.x - ego_loc.x,
                    y=cross_mid.y - ego_loc.y,
                    z=0.0,
                )
                forward_dist = to_mid.x * fwd.x + to_mid.y * fwd.y

                if forward_dist < JAYWALK_TRIGGER_DISTANCE_AHEAD:
                    walker_controller.start()
                    walker_controller.set_max_speed(WALK_SPEED)
                    walker_controller.go_to_location(walker_end_loc)
                    walker_started = True

            # ===== Chase camera (behind and above ego, aligned with ego forward view) =====
            fwd = ego_tf.get_forward_vector()
            cam_loc = ego_loc - fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        # Cleanup
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

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

        try:
            if walker_controller is not None:
                walker_controller.stop()
        except Exception:
            pass

        for actor in [walker_controller, walker, ego, camera]:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
