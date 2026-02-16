import os
import time
from queue import Queue, Empty

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA, location_to_waypoint, route_maker

load_dotenv()

# ===================== CONFIG =====================

# Networking
HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))
TM_PORT = int(os.getenv("CARLA_TM_PORT", 8000))

# Simulation
MAP_NAME = os.getenv("CARLA_MAP", "Town04")
FIXED_DELTA = 0.05
FPS = int(1.0 / FIXED_DELTA)
DURATION_SEC = 15.0

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# Ego route
EGO_SPAWN_INDEX = int(os.getenv("EGO_SPAWN_INDEX", "0"))
ROUTE_END_INDEX = int(os.getenv("ROUTE_END_INDEX", "20"))
ROUTE_XML_PATH = "/tmp/pcla_route_cut_in.xml"

# Cut-in behavior
CUTIN_SPAWN_AT_T = 2.0         # seconds after start
CUTIN_SPAWN_AHEAD_M = 25.0     # spawn ahead of ego (approx along lane)
CUTIN_FORCE_LC_AT_T = 5.0      # force lane change at this time (t seconds from start)
CUTIN_SPEED_MPS = 22.0         # desired speed
CUTIN_TM_DISTANCE_TO_LEAD = 1.0
CUTIN_IGNORE_VEHICLES_PERC = 0.0

# Chase camera (spectator) - behind & above ego, looking forward aligned with ego yaw
CAM_BACK = 8.0
CAM_UP = 3.2
CAM_PITCH = -12.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # must render for viewers
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def clamp_index(i: int, n: int) -> int:
    if n <= 0:
        return 0
    return max(0, min(i, n - 1))


def choose_spawn_points(world: carla.World, ego_idx: int, end_idx: int):
    sp = world.get_map().get_spawn_points()
    if not sp:
        raise RuntimeError("No spawn points found on this map.")

    ego_idx = clamp_index(ego_idx, len(sp))
    end_idx = clamp_index(end_idx, len(sp))
    if end_idx == ego_idx:
        end_idx = clamp_index(ego_idx + 10, len(sp))

    return sp[ego_idx], sp[end_idx]


def make_route_xml(client: carla.Client, start_tf: carla.Transform, end_tf: carla.Transform, out_path: str):
    waypoints = location_to_waypoint(client, start_tf.location, end_tf.location)
    route_maker(waypoints, out_path)


def get_driving_waypoint(map_obj: carla.Map, loc: carla.Location):
    return map_obj.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)


def safe_sensor_stop_destroy(sensor):
    if sensor is None:
        return
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        sensor.destroy()
    except Exception:
        pass


def safe_destroy(actor):
    if actor is None:
        return
    try:
        actor.destroy()
    except Exception:
        pass


def destroy_batch(client: carla.Client, actors):
    ids = []
    for a in actors:
        try:
            if a is not None:
                ids.append(a.id)
        except Exception:
            pass
    if not ids:
        return
    try:
        client.apply_batch([carla.command.DestroyActor(x) for x in ids])
    except Exception:
        pass


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = None
    tm = None

    ego = None
    cutin = None
    camera = None
    pcla = None
    video = None

    image_queue = Queue(maxsize=20)

    try:
        world = setup_world(client)
        tm = client.get_trafficmanager(TM_PORT)

        blueprints = world.get_blueprint_library()
        spectator = world.get_spectator()
        map_obj = world.get_map()

        # ---------- Spawn Ego ----------
        ego_spawn, end_spawn = choose_spawn_points(world, EGO_SPAWN_INDEX, ROUTE_END_INDEX)
        make_route_xml(client, ego_spawn, end_spawn, ROUTE_XML_PATH)

        ego_bp = blueprints.filter("vehicle.tesla.model3")
        ego_bp = ego_bp[0] if ego_bp else blueprints.filter("vehicle.*")[0]
        ego_bp.set_attribute("role_name", "hero")

        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)

        world.tick()

        # ---------- PCLA ----------
        pcla = PCLA("carl_carlv11", ego, ROUTE_XML_PATH, client)

        # ---------- Camera attached to spectator ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image) -> None:
            try:
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))
                frame_bgr = arr[:, :, :3]
                if image_queue.full():
                    try:
                        image_queue.get_nowait()
                    except Empty:
                        pass
                image_queue.put_nowait(frame_bgr)
            except Exception:
                pass

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # ---------- Scenario loop ----------
        start_sim_time = world.get_snapshot().timestamp.elapsed_seconds
        cutin_spawned = False
        cutin_forced_lc = False

        while True:
            now = world.get_snapshot().timestamp.elapsed_seconds
            t = now - start_sim_time
            if t >= DURATION_SEC:
                break

            # ===== Ego control =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            # ===== Spawn cut-in vehicle on RIGHT lane ahead =====
            if (not cutin_spawned) and t >= CUTIN_SPAWN_AT_T:
                ego_wp = get_driving_waypoint(map_obj, ego_loc)
                spawn_wp = None

                if ego_wp is not None:
                    right_wp = ego_wp.get_right_lane()
                    base_wp = ego_wp
                    if right_wp is not None and right_wp.lane_type == carla.LaneType.Driving:
                        base_wp = right_wp

                    nxt = base_wp.next(CUTIN_SPAWN_AHEAD_M)
                    if nxt:
                        spawn_wp = nxt[0]
                    else:
                        spawn_wp = base_wp

                if spawn_wp is not None:
                    spawn_tf = spawn_wp.transform
                    spawn_tf.location.z += 0.35

                    cutin_bp_list = blueprints.filter("vehicle.audi.*")
                    cutin_bp = cutin_bp_list[0] if cutin_bp_list else blueprints.filter("vehicle.*")[0]
                    cutin = world.try_spawn_actor(cutin_bp, spawn_tf)

                    if cutin is None:
                        fwd = spawn_tf.get_forward_vector()
                        spawn_tf.location = spawn_tf.location + carla.Location(x=fwd.x, y=fwd.y, z=fwd.z) * 3.0
                        cutin = world.try_spawn_actor(cutin_bp, spawn_tf)

                    if cutin is not None:
                        cutin_spawned = True
                        cutin_forced_lc = False

                        cutin.set_autopilot(True, TM_PORT)
                        tm.auto_lane_change(cutin, True)
                        tm.distance_to_leading_vehicle(cutin, float(CUTIN_TM_DISTANCE_TO_LEAD))
                        tm.ignore_vehicles_percentage(cutin, float(CUTIN_IGNORE_VEHICLES_PERC))
                        tm.set_desired_speed(cutin, float(CUTIN_SPEED_MPS))

            # ===== Force lane change from right lane into ego lane (left) =====
            if cutin is not None and (not cutin_forced_lc) and t >= CUTIN_FORCE_LC_AT_T:
                tm.force_lane_change(cutin, False)  # False -> left lane
                cutin_forced_lc = True

            # ===== Maintain speed (do not spam every tick; set occasionally) =====
            if cutin is not None:
                if int(t * 10) % 10 == 0:
                    tm.set_desired_speed(cutin, float(CUTIN_SPEED_MPS))

            # ===== Chase camera: behind & above ego, forward-facing aligned with ego yaw =====
            ego_fwd = ego_tf.get_forward_vector()
            cam_loc = ego_loc - ego_fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()

            # Drain queue; write latest frames received since last tick
            wrote_any = False
            while True:
                try:
                    frame = image_queue.get_nowait()
                except Empty:
                    break
                video.write(frame)
                wrote_any = True

            # If camera hasn't produced yet (rare at startup), keep ticking; no blocking.

        # Final flush ticks to capture last frames cleanly
        for _ in range(3):
            world.tick()
            while True:
                try:
                    video.write(image_queue.get_nowait())
                except Empty:
                    break

    finally:
        # Stop sensor callbacks BEFORE destroying actors/world settings changes
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass
        time.sleep(0.05)

        safe_sensor_stop_destroy(camera)

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

        # Use batch destroy to avoid "not found" spam/races
        destroy_batch(client, [cutin, ego])

        # Restore world settings (best-effort)
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
