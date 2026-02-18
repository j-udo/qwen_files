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

# Scenario tuning
EGO_TO_JUNCTION_MAX_DIST = 55.0
EGO_TO_JUNCTION_MIN_DIST = 25.0

ATTACKER_ROLL_START_EGO_DIST = 35.0  # when ego is this far from junction, attacker begins rolling
ATTACKER_ROLL_SPEED = 3.0            # m/s (slow roll)
ATTACKER_TARGET_LATERAL_OFFSET = 0.25  # how close to ego lane centerline the attacker tries to reach (m)
ATTACKER_HOLD_AFTER_CROSS = True
ATTACKER_HOLD_SPEED = 0.0            # stop after intruding

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


def vec2d(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def norm2(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return v, 0.0
    return v / n, n


def find_t_intersection_setup(world: carla.World):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    candidates = []
    for i, sp in enumerate(spawn_points):
        wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not wp:
            continue
        # Need an approach on a through road towards a junction
        nxts = wp.next(35.0)
        if not nxts:
            continue
        wp_ahead = nxts[0]
        if not wp_ahead.is_junction:
            # maybe junction further
            wp_cursor = wp
            for _ in range(10):
                n2 = wp_cursor.next(10.0)
                if not n2:
                    break
                wp_cursor = n2[0]
                if wp_cursor.is_junction:
                    wp_ahead = wp_cursor
                    break
        if not wp_ahead.is_junction:
            continue

        junc = wp_ahead.get_junction()
        if not junc:
            continue

        dist = wp.transform.location.distance(wp_ahead.transform.location)
        if not (EGO_TO_JUNCTION_MIN_DIST <= dist <= EGO_TO_JUNCTION_MAX_DIST):
            continue

        # Heuristic for "T": junction with relatively few driving lane waypoint pairs
        try:
            pairs = junc.get_waypoints(carla.LaneType.Driving)
        except Exception:
            continue
        if not pairs or len(pairs) > 24:
            continue

        candidates.append((len(pairs), dist, i, sp, wp, wp_ahead, junc))

    if not candidates:
        # Fallback: just pick any spawn point that leads into a junction
        for i, sp in enumerate(spawn_points):
            wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not wp:
                continue
            wp_cursor = wp
            wp_j = None
            for _ in range(12):
                n2 = wp_cursor.next(10.0)
                if not n2:
                    break
                wp_cursor = n2[0]
                if wp_cursor.is_junction:
                    wp_j = wp_cursor
                    break
            if wp_j and wp_j.get_junction():
                return i, sp, wp, wp_j, wp_j.get_junction()
        return 0, spawn_points[0], cmap.get_waypoint(spawn_points[0].location), None, None

    # Prefer smaller junctions and reasonable distance
    candidates.sort(key=lambda x: (x[0], abs(x[1] - 38.0)))
    _, _, idx, sp, wp, wp_j, junc = candidates[0]
    return idx, sp, wp, wp_j, junc


def pick_side_road_entry_for_intrusion(world: carla.World, junction: carla.Junction, ego_wp_approach: carla.Waypoint):
    cmap = world.get_map()
    ego_lane_id = ego_wp_approach.lane_id
    ego_road_id = ego_wp_approach.road_id

    pairs = junction.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return None, None

    ego_dir = ego_wp_approach.transform.get_forward_vector()
    ego_dir2 = vec2d(ego_dir.x, ego_dir.y)
    ego_dir2, _ = norm2(ego_dir2)

    best = None
    for w_in, w_out in pairs:
        if w_in.road_id == ego_road_id and w_in.lane_id == ego_lane_id:
            continue

        # We want a side road that enters roughly perpendicular or opposite-ish across the stop line
        in_dir = w_in.transform.get_forward_vector()
        in_dir2 = vec2d(in_dir.x, in_dir.y)
        in_dir2, _ = norm2(in_dir2)

        dot = float(np.dot(in_dir2, ego_dir2))
        # Perpendicular -> dot approx 0
        # Also accept somewhat angled approaches
        score_perp = abs(dot)

        # Also prefer entries whose endpoint is near ego lane centerline location inside junction
        # Use ego_wp_approach location as reference
        d = w_in.transform.location.distance(ego_wp_approach.transform.location)

        score = score_perp + 0.02 * d
        if best is None or score < best[0]:
            best = (score, w_in, w_out)

    if best is None:
        return None, None
    return best[1], best[2]


def ensure_route_file(client: carla.Client, start_loc: carla.Location, end_loc: carla.Location, route_path: str) -> str:
    if os.path.exists(route_path):
        return route_path
    waypoints = location_to_waypoint(client, start_loc, end_loc)
    route_maker(waypoints, route_path)
    return route_path


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    cmap = world.get_map()

    ego = None
    attacker = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    original_settings = world.get_settings()

    try:
        # ---------- Spawn Ego near a T-intersection ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        sp_idx, ego_spawn, ego_wp, ego_j_wp, junction = find_t_intersection_setup(world)

        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if not ego:
            # fallback
            ego_spawn = cmap.get_spawn_points()[0]
            ego = world.spawn_actor(ego_bp, ego_spawn)
            ego_wp = cmap.get_waypoint(ego_spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        world.tick()

        # Make a simple forward route for PCLA using spawn points ahead
        spawn_points = cmap.get_spawn_points()
        end_idx = min(len(spawn_points) - 1, sp_idx + 12)
        route_path = "./_pcla_route.xml"
        ensure_route_file(client, ego_spawn.location, spawn_points[end_idx].location, route_path)

        pcla = PCLA("carl_carlv11", ego, route_path, client)

        # ---------- Camera (attached to spectator, follows ego) ----------
        spectator = world.get_spectator()

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image) -> None:
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            image_queue.put(array[:, :, :3])

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # ---------- Determine attacker plan ----------
        attacker_started = False
        attacker_done = False

        attacker_entry_wp = None
        attacker_exit_wp = None
        if junction and ego_j_wp:
            attacker_entry_wp, attacker_exit_wp = pick_side_road_entry_for_intrusion(world, junction, ego_wp)

        if attacker_entry_wp is None:
            # fallback: create attacker from ego's right side and roll forward-left into lane
            attacker_entry_wp = None

        sim_start = world.get_snapshot().timestamp.elapsed_seconds

        # Spawn attacker early but keep it stopped behind stop line (approx a bit back from entry wp)
        if attacker_entry_wp is not None:
            spawn_wp = attacker_entry_wp.previous(6.0)
            if spawn_wp:
                spawn_tf = spawn_wp[0].transform
            else:
                spawn_tf = attacker_entry_wp.transform

            spawn_tf.location.z += 0.3
            attacker_bp = blueprints.filter("vehicle.audi.*")[0]
            attacker = world.try_spawn_actor(attacker_bp, spawn_tf)
            if attacker:
                attacker.set_autopilot(False)
                attacker.set_simulate_physics(True)
                attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        else:
            # Fallback spawn relative to ego
            ego_tf0 = ego.get_transform()
            fwd0 = ego_tf0.get_forward_vector()
            right0 = ego_tf0.get_right_vector()
            spawn_loc = ego_tf0.location + fwd0 * 35.0 + right0 * 9.0
            spawn_loc.z += 0.3
            spawn_rot = carla.Rotation(yaw=ego_tf0.rotation.yaw - 90.0)
            attacker_bp = blueprints.filter("vehicle.audi.*")[0]
            attacker = world.try_spawn_actor(attacker_bp, carla.Transform(spawn_loc, spawn_rot))
            if attacker:
                attacker.set_autopilot(False)
                attacker.set_simulate_physics(True)
                attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        # ---------- Main loop ----------
        while True:
            snap = world.get_snapshot()
            now = snap.timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # Ego control from PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation

            # Find distance to junction ahead (if we have it)
            ego_to_j = None
            if ego_j_wp is not None:
                ego_to_j = ego_loc.distance(ego_j_wp.transform.location)

            # Start attacker roll when ego approaches
            if attacker and not attacker_started:
                trigger = False
                if ego_to_j is not None:
                    trigger = ego_to_j < ATTACKER_ROLL_START_EGO_DIST
                else:
                    trigger = (now - sim_start) > 3.0

                if trigger:
                    attacker_started = True
                    attacker.apply_control(carla.VehicleControl(throttle=0.35, brake=0.0))
                    print("[EVENT] Attacker begins rolling past stop line")

            # Attacker motion: roll into/near ego lane
            if attacker and attacker_started and not attacker_done:
                if attacker_entry_wp is not None and ego_wp is not None:
                    # Target a point near ego approach lane centerline, slightly inside the junction
                    target_wp = ego_wp
                    if ego_j_wp is not None:
                        target_wp = ego_j_wp
                    target_loc = target_wp.transform.location

                    # Move attacker towards target_loc by setting target velocity in that direction
                    a_loc = attacker.get_location()
                    v = vec2d(target_loc.x - a_loc.x, target_loc.y - a_loc.y)
                    v_dir, v_norm = norm2(v)

                    # If close enough to ego lane centerline area, stop (or keep slight roll)
                    if v_norm < 3.5:
                        attacker_done = True
                        if ATTACKER_HOLD_AFTER_CROSS:
                            attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        else:
                            attacker.set_target_velocity(carla.Vector3D(v_dir[0] * 0.5, v_dir[1] * 0.5, 0.0))
                        print("[EVENT] Attacker has intruded into/near ego lane")
                    else:
                        attacker.set_target_velocity(carla.Vector3D(v_dir[0] * ATTACKER_ROLL_SPEED, v_dir[1] * ATTACKER_ROLL_SPEED, 0.0))
                else:
                    # Fallback: just cross leftwards relative to ego heading
                    right = ego_tf.get_right_vector()
                    cross = carla.Vector3D(x=-right.x, y=-right.y, z=0.0)
                    attacker.set_target_velocity(cross * ATTACKER_ROLL_SPEED)

            # Chase camera: slightly above and behind ego, aligned with ego forward view
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_loc - forward * 7.5 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-10.0, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # Tick & record
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
            if video:
                video.release()
                print(f"[INFO] Video saved: {VIDEO_PATH}")
        except Exception:
            pass

        try:
            if pcla:
                pcla.cleanup()
        except Exception:
            pass

        for actor in [attacker, ego]:
            try:
                if actor:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            settings.no_rendering_mode = original_settings.no_rendering_mode
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
