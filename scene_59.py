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

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# PCLA
AGENT_NAME = "carl_carlv11"
ROUTE_XML = "./sample_route.xml"

# Scenario timing
TOTAL_DURATION_SEC = 15.0

# Lead truck + reveal setup
TRUCK_AHEAD_DISTANCE = 16.0      # meters ahead of ego at spawn
TRUCK_LANE_CHANGE_TIME = 7.0     # seconds into scenario when truck moves aside
TRUCK_LANE_CHANGE_DURATION = 2.0 # seconds to complete lateral shift
TRUCK_LATERAL_SHIFT = 3.6        # meters lateral shift to adjacent lane-ish

# Hidden/stopped vehicle ahead (in ego lane, initially occluded by truck)
STOPPED_AHEAD_DISTANCE = 28.0    # meters ahead of ego at spawn
STOPPED_VEHICLE_Z = 0.25

# Keep traffic actors from moving
FREEZE_VELOCITY = carla.Vector3D(0.0, 0.0, 0.0)

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


def pick_blueprint(blueprints: carla.BlueprintLibrary, patterns, fallback="vehicle.carlamotors.carlacola") -> carla.ActorBlueprint:
    for pat in patterns:
        bps = blueprints.filter(pat)
        if bps:
            return bps[0]
    bp = blueprints.find(fallback) if blueprints.find(fallback) else None
    if bp:
        return bp
    # last resort: any vehicle
    return blueprints.filter("vehicle.*")[0]


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    m = world.get_map()
    blueprints = world.get_blueprint_library()

    ego = None
    pcla = None
    camera = None
    video = None
    truck = None
    stopped = None

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0] if blueprints.filter("vehicle.tesla.model3") else blueprints.filter("vehicle.*")[0]
        spawn_points = m.get_spawn_points()
        ego_spawn = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        start_sim_t = world.get_snapshot().timestamp.elapsed_seconds

        # ---------- PCLA ----------
        pcla = PCLA(AGENT_NAME, ego, ROUTE_XML, client)

        # ---------- Camera attached behind/above ego ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        # Slightly above and behind the ego, forward-facing aligned with ego (relative transform)
        cam_rel_tf = carla.Transform(
            carla.Location(x=-7.0, z=3.0),
            carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_rel_tf, attach_to=ego)

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

        # ---------- Spawn lead truck and hidden stopped vehicle ----------
        ego_tf0 = ego.get_transform()
        forward0 = ego_tf0.get_forward_vector()
        right0 = ego_tf0.get_right_vector()

        truck_bp = pick_blueprint(
            blueprints,
            patterns=[
                "vehicle.carlamotors.carlacola",
                "vehicle.carlamotors.firetruck",
                "vehicle.*truck*",
                "vehicle.*carlacola*",
            ],
        )
        truck_spawn_loc = ego_tf0.location + forward0 * TRUCK_AHEAD_DISTANCE
        truck_spawn_loc.z += 0.25
        truck_tf = carla.Transform(truck_spawn_loc, ego_tf0.rotation)
        truck = world.try_spawn_actor(truck_bp, truck_tf)

        stopped_bp = pick_blueprint(
            blueprints,
            patterns=[
                "vehicle.*",
            ],
        )
        stopped_spawn_loc = ego_tf0.location + forward0 * STOPPED_AHEAD_DISTANCE
        stopped_spawn_loc.z += STOPPED_VEHICLE_Z
        stopped_tf = carla.Transform(stopped_spawn_loc, ego_tf0.rotation)
        stopped = world.try_spawn_actor(stopped_bp, stopped_tf)

        if truck:
            truck.set_autopilot(False)
            truck.set_simulate_physics(True)

        if stopped:
            stopped.set_autopilot(False)
            stopped.set_simulate_physics(True)

        lane_change_started = False
        lane_change_start_t = None
        truck_start_loc = None

        # ---------- Main loop ----------
        while True:
            snap = world.get_snapshot()
            now_t = snap.timestamp.elapsed_seconds
            elapsed = now_t - start_sim_t
            if elapsed >= TOTAL_DURATION_SEC:
                break

            # Ego control via PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            # Keep truck and stopped car mostly stationary in longitudinal direction
            if stopped:
                stopped.set_target_velocity(FREEZE_VELOCITY)
            if truck and (not lane_change_started):
                truck.set_target_velocity(FREEZE_VELOCITY)

            # Truck lane change (lateral shift) to reveal stopped vehicle
            if truck and (elapsed >= TRUCK_LANE_CHANGE_TIME):
                if not lane_change_started:
                    lane_change_started = True
                    lane_change_start_t = now_t
                    truck_start_loc = truck.get_location()

                phase = (now_t - lane_change_start_t) / max(TRUCK_LANE_CHANGE_DURATION, 0.001)
                if phase > 1.0:
                    phase = 1.0

                # Use current ego right vector for consistent lateral direction in camera view
                ego_tf = ego.get_transform()
                right = ego_tf.get_right_vector()

                # Shift truck to the right (adjacent lane) smoothly
                target_loc = carla.Location(
                    x=truck_start_loc.x + right.x * TRUCK_LATERAL_SHIFT * phase,
                    y=truck_start_loc.y + right.y * TRUCK_LATERAL_SHIFT * phase,
                    z=truck_start_loc.z,
                )

                truck_tf_now = truck.get_transform()
                new_tf = carla.Transform(target_loc, truck_tf_now.rotation)
                truck.set_transform(new_tf)
                truck.set_target_velocity(FREEZE_VELOCITY)

            # Tick & record
            world.tick()

            while not image_queue.empty():
                frame = image_queue.get()
                video.write(frame)

    finally:
        try:
            if camera:
                camera.stop()
        except Exception:
            pass

        try:
            if video:
                video.release()
        except Exception:
            pass

        try:
            if pcla:
                pcla.cleanup()
        except Exception:
            pass

        for actor in [camera, truck, stopped, ego]:
            try:
                if actor:
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
