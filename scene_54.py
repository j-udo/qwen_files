import os
import time
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA
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

# Ego / PCLA
AGENT_NAME = "carl_carlv11"
ROUTE_XML = "./sample_route.xml"

# Cut-in behavior (attacker from right lane cuts into ego lane ahead)
EGO_PROGRESS_TRIGGER = 10.0   # meters from start before spawning attacker
CUTIN_AHEAD_DIST = 18.0       # attacker spawn distance ahead of ego (meters)
RIGHT_LANE_OFFSET = 3.6       # approximate lane width offset to the right (meters)
CUTIN_DURATION = 2.2          # seconds to complete lateral cut-in
POST_CUTIN_HOLD = 4.0         # keep driving after cut-in

# Camera (chase view)
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


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    carla_map = world.get_map()

    ego = None
    attacker = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        ego.set_autopilot(False)
        ego.set_simulate_physics(True)

        start_loc = ego.get_location()
        world.tick()

        # ---------- PCLA ----------
        pcla = PCLA(AGENT_NAME, ego, ROUTE_XML, client)

        # ---------- Spectator / Camera ----------
        spectator = world.get_spectator()

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

        # ---------- Timing ----------
        sim_start = world.get_snapshot().timestamp.elapsed_seconds

        # ---------- Attacker state ----------
        attacker_spawned = False
        cutin_started_time = None
        attacker_start_loc = None
        attacker_target_loc = None
        attacker_forward_speed = 0.0
        attacker_yaw = None
        attacker_lane_change_dir = None  # vector pointing left (towards ego lane)

        while True:
            snapshot = world.get_snapshot()
            sim_now = snapshot.timestamp.elapsed_seconds
            if sim_now - sim_start >= DURATION_SECONDS:
                break

            # ===== Ego control (PCLA) =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            ego_forward = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()

            # ===== Spawn attacker in right lane ahead =====
            ego_progress = ego_loc.distance(start_loc)
            if (not attacker_spawned) and ego_progress >= EGO_PROGRESS_TRIGGER:
                # Put attacker ahead and to the right (right lane), aligned with ego heading
                spawn_loc = ego_loc + ego_forward * CUTIN_AHEAD_DIST + ego_right * RIGHT_LANE_OFFSET
                spawn_loc.z += 0.3

                attacker_bp_candidates = blueprints.filter("vehicle.audi.*")
                attacker_bp = attacker_bp_candidates[0] if attacker_bp_candidates else blueprints.filter("vehicle.*")[0]

                attacker_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_rot.yaw))
                attacker = world.try_spawn_actor(attacker_bp, attacker_tf)

                if attacker is not None:
                    attacker.set_autopilot(False)
                    attacker.set_simulate_physics(True)

                    # Set initial forward speed similar to ego's current speed (fallback)
                    ego_vel = ego.get_velocity()
                    ego_speed = float((ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2) ** 0.5)
                    attacker_forward_speed = max(6.0, ego_speed + 1.5)

                    attacker_spawned = True
                    cutin_started_time = None
                    attacker_start_loc = None
                    attacker_target_loc = None
                    attacker_yaw = ego_rot.yaw
                    attacker_lane_change_dir = carla.Vector3D(x=-ego_right.x, y=-ego_right.y, z=0.0)

            # ===== Attacker behavior: right-lane cut-in to ego lane =====
            if attacker_spawned and attacker is not None:
                if cutin_started_time is None:
                    cutin_started_time = sim_now
                    attacker_start_loc = attacker.get_location()
                    attacker_target_loc = attacker_start_loc + carla.Location(
                        x=attacker_lane_change_dir.x * RIGHT_LANE_OFFSET,
                        y=attacker_lane_change_dir.y * RIGHT_LANE_OFFSET,
                        z=0.0,
                    )

                t = (sim_now - cutin_started_time) / max(0.001, CUTIN_DURATION)
                t = clamp(t, 0.0, 1.0)

                # Lateral interpolate (teleport for deterministic visible cut-in), keep yaw aligned with road.
                new_loc = carla.Location(
                    x=attacker_start_loc.x + (attacker_target_loc.x - attacker_start_loc.x) * t,
                    y=attacker_start_loc.y + (attacker_target_loc.y - attacker_start_loc.y) * t,
                    z=attacker_start_loc.z,
                )
                attacker.set_transform(carla.Transform(new_loc, carla.Rotation(yaw=attacker_yaw)))

                # Keep it moving forward after/during cut-in.
                attacker.set_target_velocity(carla.Vector3D(
                    x=ego_forward.x * attacker_forward_speed,
                    y=ego_forward.y * attacker_forward_speed,
                    z=0.0
                ))

                # After cut-in, just keep pace for a bit (velocity already set), then nothing special.

            # ===== Camera: slightly above and behind ego, forward-facing aligned =====
            cam_loc = ego_loc - ego_forward * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()
            while not image_queue.empty():
                video.write(image_queue.get())

            # Small guard to ensure some frames even if sensor lags
            time.sleep(0.0)

    finally:
        # Cleanup sensors/video first
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        try:
            if camera is not None:
                camera.destroy()
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

        for actor in [attacker, ego]:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
