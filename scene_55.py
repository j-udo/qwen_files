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

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
DURATION_SECONDS = 15.0

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

AGENT_NAME = "carl_carlv11"
ROUTE_XML = "./sample_route.xml"

# Scenario: vehicle in right lane cuts in front of ego
SPAWN_AFTER_SECONDS = 2.0          # ensure ego is moving before spawning
CUTIN_TARGET_GAP_AHEAD = 10.0      # attacker ends up ~this far ahead when merged (meters)
ATTACKER_SPAWN_GAP_AHEAD = 18.0    # attacker initially ahead in right lane (meters)
CUTIN_START_DELAY = 1.0           # seconds after spawn to begin lateral merge
CUTIN_DURATION = 2.2              # seconds to merge laterally
ATTACKER_SPEED_DELTA = 3.0        # attacker speed relative to ego (m/s), to make cut-in apparent

# Camera: chase view behind/above ego, aligned with ego yaw
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


def speed_mps(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def try_get_right_lane_waypoint(carla_map: carla.Map, ego_wp: carla.Waypoint) -> carla.Waypoint:
    if ego_wp is None:
        return None
    rw = ego_wp.get_right_lane()
    if rw is not None and rw.lane_type == carla.LaneType.Driving:
        return rw
    return None


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

    attacker_spawn_time = None
    cutin_begin_time = None
    attacker_right_wp = None
    attacker_left_wp = None
    attacker_route_start_s = None  # for reference only

    try:
        # ---------- Spawn Ego ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        ego.set_autopilot(False)
        ego.set_simulate_physics(True)
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

        sim_start = world.get_snapshot().timestamp.elapsed_seconds

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
            ego_fwd = ego_tf.get_forward_vector()

            # ===== Determine lanes using waypoints =====
            ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            right_wp = try_get_right_lane_waypoint(carla_map, ego_wp)

            # ===== Spawn attacker in actual right lane ahead =====
            if attacker is None and (sim_now - sim_start) >= SPAWN_AFTER_SECONDS and right_wp is not None:
                # Choose a point ahead on ego lane then move to the right lane waypoint at that s-position
                ahead_wps = ego_wp.next(ATTACKER_SPAWN_GAP_AHEAD) if ego_wp is not None else []
                if ahead_wps:
                    ego_ahead_wp = ahead_wps[0]
                    right_ahead_wp = try_get_right_lane_waypoint(carla_map, ego_ahead_wp)
                else:
                    right_ahead_wp = None

                if right_ahead_wp is None:
                    # fallback to right lane at current position (less ideal but still visible)
                    right_ahead_wp = right_wp

                spawn_tf = carla.Transform(
                    right_ahead_wp.transform.location + carla.Location(z=0.3),
                    right_ahead_wp.transform.rotation,
                )

                attacker_bp_candidates = blueprints.filter("vehicle.audi.*")
                attacker_bp = attacker_bp_candidates[0] if attacker_bp_candidates else blueprints.filter("vehicle.*")[0]

                attacker = world.try_spawn_actor(attacker_bp, spawn_tf)
                if attacker is not None:
                    attacker.set_autopilot(False)
                    attacker.set_simulate_physics(True)
                    attacker_spawn_time = sim_now
                    cutin_begin_time = None

                    # Cache target lane waypoints at similar longitudinal progress
                    attacker_right_wp = right_ahead_wp
                    attacker_left_wp = carla_map.get_waypoint(
                        ego_ahead_wp.transform.location if ahead_wps else ego_wp.transform.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )

            # ===== Attacker behavior: maintain speed, then cut left into ego lane in front =====
            if attacker is not None:
                ego_spd = speed_mps(ego)
                desired_attacker_spd = max(8.0, ego_spd + ATTACKER_SPEED_DELTA)

                # Keep attacker moving forward aligned with its current transform
                att_tf = attacker.get_transform()
                att_fwd = att_tf.get_forward_vector()
                attacker.set_target_velocity(
                    carla.Vector3D(att_fwd.x * desired_attacker_spd, att_fwd.y * desired_attacker_spd, 0.0)
                )

                # Decide when to start cut-in
                if cutin_begin_time is None and attacker_spawn_time is not None:
                    if (sim_now - attacker_spawn_time) >= CUTIN_START_DELAY:
                        cutin_begin_time = sim_now

                        # Recompute "merge-to" location ~CUTIN_TARGET_GAP_AHEAD ahead of ego on ego lane
                        ego_wp_now = carla_map.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                        merge_wps = ego_wp_now.next(CUTIN_TARGET_GAP_AHEAD) if ego_wp_now is not None else []
                        merge_wp = merge_wps[0] if merge_wps else ego_wp_now

                        # And "from" location: attacker current lane center (right lane)
                        from_wp = carla_map.get_waypoint(attacker.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)

                        attacker_right_wp = from_wp
                        attacker_left_wp = merge_wp

                # Execute cut-in via smooth transform interpolation between lane centers
                if cutin_begin_time is not None and attacker_right_wp is not None and attacker_left_wp is not None:
                    t = (sim_now - cutin_begin_time) / max(0.001, CUTIN_DURATION)
                    t = clamp(t, 0.0, 1.0)

                    from_loc = attacker_right_wp.transform.location
                    to_loc = attacker_left_wp.transform.location

                    # Keep some forward motion component by blending with current location as well
                    # but primary effect is a clear right->left lane change in front of ego.
                    new_loc = carla.Location(
                        x=from_loc.x + (to_loc.x - from_loc.x) * t,
                        y=from_loc.y + (to_loc.y - from_loc.y) * t,
                        z=max(from_loc.z, to_loc.z) + 0.3,
                    )

                    # Use road-aligned yaw from target (ego lane) so it looks like a real merge
                    new_yaw = attacker_left_wp.transform.rotation.yaw
                    attacker.set_transform(carla.Transform(new_loc, carla.Rotation(yaw=new_yaw)))

            # ===== Camera: behind/above ego, forward-facing aligned =====
            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            ego_fwd = ego_tf.get_forward_vector()

            cam_loc = ego_loc - ego_fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()
            while not image_queue.empty():
                video.write(image_queue.get())

            time.sleep(0.0)

    finally:
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
