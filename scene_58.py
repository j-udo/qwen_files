import os
import time
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

# Ego / Route
EGO_SPAWN_INDEX = 31
EGO_ROUTE_END_INDEX = 42
ROUTE_XML_PATH = "/tmp/pcla_route.xml"

# Oncoming vehicle / U-turn behavior
ONCOMING_SPAWN_AHEAD = 55.0          # meters ahead of ego, in opposite lane
UTURN_TRIGGER_DISTANCE = 35.0        # begin turning when ego is within this distance
UTURN_DURATION = 3.0                 # seconds to complete aggressive U-turn
ONCOMING_SPEED = 10.0                # m/s (moderate approach speed)
UTURN_FORWARD_SPEED = 8.0            # m/s during turn (keeps it moving while rotating)
MAX_YAW_RATE = 160.0                 # deg/s during U-turn (aggressive)

# Camera (spectator-follow)
CAM_BACK = 7.0
CAM_UP = 3.2
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


def _vec2d_norm(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _safe_destroy(actor):
    try:
        if actor is not None:
            actor.destroy()
    except Exception:
        pass


def _safe_stop_destroy_sensor(sensor):
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    _safe_destroy(sensor)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    cmap = world.get_map()

    # ---------- Spawn Ego ----------
    spawn_points = cmap.get_spawn_points()
    ego_spawn = spawn_points[min(EGO_SPAWN_INDEX, len(spawn_points) - 1)]

    ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
    ego_bp.set_attribute("role_name", "hero")

    ego = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego is None:
        # Fallback to any available spawn
        for sp in spawn_points:
            ego = world.try_spawn_actor(ego_bp, sp)
            if ego is not None:
                break
    if ego is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    world.tick()

    # ---------- Create a simple straight-ish route XML for PCLA ----------
    end_spawn = spawn_points[min(EGO_ROUTE_END_INDEX, len(spawn_points) - 1)]
    try:
        wps = location_to_waypoint(client, ego_spawn.location, end_spawn.location)
        route_maker(wps, ROUTE_XML_PATH)
    except Exception:
        # If route generation fails for any reason, still try to run with a known sample path.
        ROUTE_XML_PATH = "./sample_route.xml"

    # ---------- PCLA ----------
    pcla = PCLA("carl_carlv11", ego, ROUTE_XML_PATH, client)

    # ---------- Spectator / Camera sensor attached to spectator ----------
    spectator = world.get_spectator()

    cam_bp = blueprints.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

    image_queue: Queue[np.ndarray] = Queue()

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

    # ---------- Spawn oncoming vehicle in opposite lane ahead ----------
    oncoming = None
    uturn_started = False
    uturn_start_time = None

    try:
        ego_tf0 = ego.get_transform()
        ego_wp0 = cmap.get_waypoint(ego_tf0.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp0 is None:
            raise RuntimeError("Could not get ego waypoint.")

        # Find opposite lane waypoint (oncoming)
        opposite_wp = ego_wp0.get_left_lane()
        if opposite_wp is None or opposite_wp.lane_type != carla.LaneType.Driving:
            opposite_wp = ego_wp0.get_right_lane()
        if opposite_wp is None or opposite_wp.lane_type != carla.LaneType.Driving:
            opposite_wp = ego_wp0  # fallback (still spawn ahead)

        # Move ahead along road
        ahead_candidates = opposite_wp.next(ONCOMING_SPAWN_AHEAD)
        if not ahead_candidates:
            ahead_candidates = opposite_wp.next(20.0)
        oncoming_wp = ahead_candidates[0] if ahead_candidates else opposite_wp

        # Make it face toward ego (reverse direction)
        oncoming_tf = carla.Transform(
            oncoming_wp.transform.location + carla.Location(z=0.3),
            carla.Rotation(yaw=oncoming_wp.transform.rotation.yaw + 180.0)
        )

        oncoming_bp = blueprints.filter("vehicle.*")[0]
        for cand in ["vehicle.audi.a2", "vehicle.bmw.grandtourer", "vehicle.toyota.prius", "vehicle.lincoln.mkz_2017"]:
            found = blueprints.filter(cand)
            if found:
                oncoming_bp = found[0]
                break
        oncoming_bp.set_attribute("role_name", "scenario")

        oncoming = world.try_spawn_actor(oncoming_bp, oncoming_tf)
        if oncoming is not None:
            oncoming.set_autopilot(False)
            oncoming.set_simulate_physics(True)

        world.tick()

        start_wall = time.time()
        while True:
            elapsed = time.time() - start_wall
            if elapsed >= DURATION_SECONDS:
                break

            # ===== Ego control via PCLA =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw = ego_tf.rotation.yaw

            # ===== Oncoming approach & U-turn =====
            if oncoming is not None:
                onc_tf = oncoming.get_transform()
                onc_loc = onc_tf.location

                dist = ego_loc.distance(onc_loc)

                if (not uturn_started) and dist < UTURN_TRIGGER_DISTANCE:
                    uturn_started = True
                    uturn_start_time = world.get_snapshot().timestamp.elapsed_seconds

                if not uturn_started:
                    # Approach ego in its lane (toward ego): drive in its forward direction (which is opposite road direction)
                    fwd = onc_tf.get_forward_vector()
                    oncoming.set_target_velocity(carla.Vector3D(fwd.x * ONCOMING_SPEED, fwd.y * ONCOMING_SPEED, 0.0))
                    oncoming.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                else:
                    now_sim = world.get_snapshot().timestamp.elapsed_seconds
                    t = max(0.0, now_sim - uturn_start_time)
                    alpha = min(1.0, t / max(UTURN_DURATION, 0.01))

                    # Rotate aggressively by ~180 degrees during the turn
                    yaw_rate = MAX_YAW_RATE
                    if alpha > 0.95:
                        yaw_rate = MAX_YAW_RATE * (1.0 - alpha) * 20.0

                    # Keep moving forward while rotating to simulate sweeping U-turn across ego lane
                    fwd = onc_tf.get_forward_vector()
                    oncoming.set_target_velocity(carla.Vector3D(fwd.x * UTURN_FORWARD_SPEED, fwd.y * UTURN_FORWARD_SPEED, 0.0))

                    # Determine turn direction to cut across ego lane (choose direction based on relative heading)
                    # We want it to swing across the centerline; pick a consistent sign.
                    oncoming.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, yaw_rate))

            # ===== Chase camera (slightly above and behind ego, forward-facing aligned with ego) =====
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_loc - forward * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        _safe_stop_destroy_sensor(camera)
        try:
            video.release()
        except Exception:
            pass

        try:
            pcla.cleanup()
        except Exception:
            pass

        _safe_destroy(oncoming)
        _safe_destroy(ego)

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
