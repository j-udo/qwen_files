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

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

DURATION_SECONDS = 15.0

# Scenario behavior
TRACKS_AHEAD_DISTANCE = 55.0          # approx location of crossing center ahead of start
WARNING_TRIGGER_DISTANCE = 35.0       # when ego within this distance, lights activate
BARRIER_CLOSE_DELAY = 1.5             # seconds after lights activate, barrier blocks lane

LIGHT_POST_DISTANCE_BEFORE_TRACKS = 10.0
LIGHT_LATERAL_OFFSET = 4.0

BARRIER_LATERAL_OFFSET = 2.2
BARRIER_Z_OFFSET = 0.6

DEBUG_DRAW = True

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    # Prefer using current world if already correct map; load only if needed.
    world = client.get_world()
    try:
        current_map_name = world.get_map().name
    except Exception:
        current_map_name = ""

    if MAP_NAME not in current_map_name:
        print(f"[INFO] Loading map: {MAP_NAME} (current: {current_map_name})")
        world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # ensure rendered
    world.apply_settings(settings)

    # If TrafficManager is running, keep it in sync to avoid stalls.
    try:
        tm = client.get_trafficmanager(int(os.getenv("CARLA_TM_PORT", "8000")))
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world


def _vec2(v: carla.Vector3D) -> carla.Vector3D:
    return carla.Vector3D(v.x, v.y, 0.0)


def _norm2(v: carla.Vector3D) -> float:
    return float((v.x * v.x + v.y * v.y) ** 0.5)


def _unit2(v: carla.Vector3D) -> carla.Vector3D:
    n = _norm2(v)
    if n < 1e-6:
        return carla.Vector3D(1.0, 0.0, 0.0)
    return carla.Vector3D(v.x / n, v.y / n, 0.0)


def _set_vehicle_emergency_flashers(vehicle: carla.Vehicle, on: bool) -> None:
    try:
        if on:
            vehicle.set_light_state(
                carla.VehicleLightState(
                    carla.VehicleLightState.Emergency | carla.VehicleLightState.Position
                )
            )
        else:
            vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.NONE))
    except Exception:
        pass


def _pick_vehicle_bp(bp_lib: carla.BlueprintLibrary, pattern_list) -> carla.ActorBlueprint:
    for patt in pattern_list:
        bps = bp_lib.filter(patt)
        if bps:
            return bps[0]
    bps = bp_lib.filter("vehicle.*")
    return bps[0]


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = setup_world(client)
    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()

    # ---------- Spawn Ego ----------
    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found on map")

    ego_spawn_idx = 31 if len(spawn_points) > 31 else 0
    ego_spawn = spawn_points[ego_spawn_idx]

    ego_bp = _pick_vehicle_bp(bp_lib, ["vehicle.tesla.model3", "vehicle.*"])
    try:
        ego_bp.set_attribute("role_name", "hero")
    except Exception:
        pass

    ego = world.try_spawn_actor(ego_bp, ego_spawn)
    if ego is None:
        ego = world.spawn_actor(ego_bp, ego_spawn)
    world.tick()

    # ---------- PCLA ----------
    # Use the known-good sample route from the example.
    route_xml = "./sample_route.xml"
    pcla = PCLA("carl_carlv11", ego, route_xml, client)
    print("[INFO] Ego spawned, PCLA running")

    # ---------- Spectator / Camera ----------
    spectator = world.get_spectator()

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(IMAGE_W))
    cam_bp.set_attribute("image_size_y", str(IMAGE_H))
    cam_bp.set_attribute("fov", str(FOV))

    camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

    image_queue: Queue = Queue()

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

    # ---------- Precompute "tracks" and stopline reference points ----------
    ego_start_tf = ego.get_transform()
    forward0 = _unit2(_vec2(ego_start_tf.get_forward_vector()))
    right0 = _unit2(_vec2(ego_start_tf.get_right_vector()))

    tracks_center = ego_start_tf.location + carla.Location(
        x=forward0.x * TRACKS_AHEAD_DISTANCE,
        y=forward0.y * TRACKS_AHEAD_DISTANCE,
        z=0.0,
    )
    tracks_wp = carla_map.get_waypoint(tracks_center, project_to_road=True, lane_type=carla.LaneType.Driving)
    if tracks_wp is not None:
        tracks_center = tracks_wp.transform.location

    stopline_loc = tracks_center - carla.Location(
        x=forward0.x * 8.0,
        y=forward0.y * 8.0,
        z=0.0,
    )
    stopline_wp = carla_map.get_waypoint(stopline_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if stopline_wp is not None:
        stopline_loc = stopline_wp.transform.location

    # ---------- Warning lights "posts" (frozen vehicles with flashers) ----------
    warning_posts = []
    warning_posts_active = False

    def spawn_warning_post(side_sign: float) -> carla.Vehicle:
        loc = tracks_center - carla.Location(
            x=forward0.x * LIGHT_POST_DISTANCE_BEFORE_TRACKS,
            y=forward0.y * LIGHT_POST_DISTANCE_BEFORE_TRACKS,
            z=0.0,
        )
        loc = loc + carla.Location(
            x=right0.x * LIGHT_LATERAL_OFFSET * side_sign,
            y=right0.y * LIGHT_LATERAL_OFFSET * side_sign,
            z=0.0,
        )
        loc.z += 0.3

        rot = carla.Rotation(yaw=ego_start_tf.rotation.yaw, pitch=0.0, roll=0.0)
        tf = carla.Transform(loc, rot)

        bp = _pick_vehicle_bp(bp_lib, ["vehicle.micro.microlino", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "static_warning_post")
        except Exception:
            pass

        v = world.try_spawn_actor(bp, tf)
        if v:
            try:
                v.set_simulate_physics(False)
                v.set_autopilot(False)
            except Exception:
                pass
            _set_vehicle_emergency_flashers(v, False)
        return v

    post_left = spawn_warning_post(+1.0)
    post_right = spawn_warning_post(-1.0)
    if post_left:
        warning_posts.append(post_left)
    if post_right:
        warning_posts.append(post_right)

    # ---------- Barrier (blocking vehicle moved into lane) ----------
    barrier_vehicle = None
    barrier_raised_tf = None
    barrier_lowered_tf = None
    barrier_active = False
    barrier_close_time = None

    def spawn_barrier() -> None:
        nonlocal barrier_vehicle, barrier_raised_tf, barrier_lowered_tf

        bp = _pick_vehicle_bp(bp_lib, ["vehicle.carlamotors.carlacola", "vehicle.ford.ambulance", "vehicle.*"])
        try:
            bp.set_attribute("role_name", "rr_barrier")
        except Exception:
            pass

        yaw_across = ego_start_tf.rotation.yaw + 90.0

        base_loc = stopline_loc + carla.Location(
            x=right0.x * BARRIER_LATERAL_OFFSET,
            y=right0.y * BARRIER_LATERAL_OFFSET,
            z=0.0,
        )
        base_loc.z += BARRIER_Z_OFFSET

        raised_loc = base_loc + carla.Location(
            x=right0.x * 2.5,
            y=right0.y * 2.5,
            z=0.0,
        )

        barrier_raised_tf = carla.Transform(raised_loc, carla.Rotation(yaw=yaw_across))
        barrier_lowered_tf = carla.Transform(base_loc, carla.Rotation(yaw=yaw_across))

        barrier_vehicle = world.try_spawn_actor(bp, barrier_raised_tf)
        if barrier_vehicle:
            try:
                barrier_vehicle.set_simulate_physics(False)
                barrier_vehicle.set_autopilot(False)
            except Exception:
                pass

    spawn_barrier()

    # ---------- Debug visuals ----------
    debug = world.debug

    def draw_debug():
        if not DEBUG_DRAW:
            return
        try:
            debug.draw_point(tracks_center + carla.Location(z=0.5), size=0.18, color=carla.Color(255, 0, 0), life_time=0.11)
            debug.draw_point(stopline_loc + carla.Location(z=0.5), size=0.18, color=carla.Color(255, 255, 0), life_time=0.11)
            debug.draw_line(
                stopline_loc + carla.Location(z=0.4),
                stopline_loc + carla.Location(z=0.4) + carla.Location(x=right0.x * 4.0, y=right0.y * 4.0),
                thickness=0.08,
                color=carla.Color(0, 255, 255),
                life_time=0.11,
            )
        except Exception:
            pass

    # ---------- Main loop (15 seconds) ----------
    start_wall = time.time()
    rr_lights_start_simtime = None

    try:
        while True:
            wall_elapsed = time.time() - start_wall
            if wall_elapsed >= DURATION_SECONDS:
                print("[INFO] Terminating scenario (15s reached)")
                break

            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds

            # ===== Ego control via PCLA =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location

            # Distance to tracks center
            dist_to_tracks = ego_loc.distance(tracks_center)

            # ===== Trigger warning sequence =====
            if rr_lights_start_simtime is None and dist_to_tracks < WARNING_TRIGGER_DISTANCE:
                rr_lights_start_simtime = sim_time
                barrier_close_time = rr_lights_start_simtime + BARRIER_CLOSE_DELAY
                warning_posts_active = True
                print("[EVENT] Railroad warning lights activated")

            # Flash warning lights (alternating every 0.4s)
            if warning_posts_active and warning_posts and rr_lights_start_simtime is not None:
                phase = int((sim_time - rr_lights_start_simtime) / 0.4) % 2
                for i, post in enumerate(warning_posts):
                    if not post:
                        continue
                    on = (phase == 0) if (i % 2 == 0) else (phase == 1)
                    _set_vehicle_emergency_flashers(post, on)

            # Lower barrier after delay (teleport blocker into lane)
            if barrier_vehicle and rr_lights_start_simtime is not None and not barrier_active:
                if barrier_close_time is not None and sim_time >= barrier_close_time:
                    barrier_active = True
                    try:
                        barrier_vehicle.set_transform(barrier_lowered_tf)
                    except Exception:
                        pass
                    print("[EVENT] Railroad barrier lowered (blocking lane)")

            # ===== Camera: above and behind ego, aligned forward =====
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_loc - forward * 7.0 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-10.0, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            draw_debug()

            # ===== Tick & record =====
            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        print("[INFO] Cleaning up")

        try:
            if camera:
                camera.stop()
                camera.destroy()
        except Exception:
            pass

        try:
            video.release()
            print(f"[INFO] Video saved: {VIDEO_PATH}")
        except Exception:
            pass

        try:
            pcla.cleanup()
        except Exception:
            pass

        for actor in [barrier_vehicle] + warning_posts + [ego]:
            try:
                if actor:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        try:
            tm = client.get_trafficmanager(int(os.getenv("CARLA_TM_PORT", "8000")))
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
