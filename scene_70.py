import os
import math
from queue import Queue

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA, location_to_waypoint, route_maker

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1.0 / FIXED_DELTA)

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

DURATION_SECONDS = 15.0

# Ego approach setup
EGO_TO_JUNCTION_MIN = 25.0
EGO_TO_JUNCTION_MAX = 60.0

# Attacker behavior: slow roll past stop line into ego lane
TRIGGER_EGO_TO_JUNCTION = 18.0  # start rolling when ego is close to the junction
ATTACKER_SPEED = 2.0            # m/s slow roll
ATTACKER_STOP_AFTER_INTRUDE = True
ATTACKER_INTRUDE_DEPTH = 2.2    # meters past the stop line into junction/ego lane vicinity (approx)


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # must render
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def vec2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def norm2(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return v, 0.0
    return v / n, n


def angle_between_deg(a, b):
    a, _ = norm2(a)
    b, _ = norm2(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def advance_to_junction(wp: carla.Waypoint, max_dist=120.0, step=6.0):
    dist = 0.0
    cur = wp
    while dist < max_dist:
        nxt = cur.next(step)
        if not nxt:
            return None, None
        cur = nxt[0]
        dist += step
        if cur.is_junction:
            return cur, dist
    return None, None


def find_t_intersection_ego_spawn(world: carla.World):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    best = None
    for i, sp in enumerate(spawn_points):
        wp0 = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            continue

        j_wp, d_to_j = advance_to_junction(wp0, max_dist=140.0, step=7.0)
        if j_wp is None:
            continue
        if not (EGO_TO_JUNCTION_MIN <= d_to_j <= EGO_TO_JUNCTION_MAX):
            continue

        junc = j_wp.get_junction()
        if junc is None:
            continue

        pairs = junc.get_waypoints(carla.LaneType.Driving)
        if not pairs:
            continue

        # Heuristic for T-ish: fewer lane-pairs than a big 4-way
        if len(pairs) > 26:
            continue

        # Ensure ego approach has a "straight-ish" continuation through the junction (avoid forced turns).
        # Choose the option at ~18m ahead that best matches current heading.
        ego_fwd = wp0.transform.get_forward_vector()
        ego_dir = vec2(ego_fwd.x, ego_fwd.y)
        ego_dir, _ = norm2(ego_dir)

        wp_before_j = wp0
        for _ in range(6):
            nxt = wp_before_j.next(6.0)
            if not nxt:
                break
            if nxt[0].is_junction:
                break
            wp_before_j = nxt[0]

        options = wp_before_j.next(18.0)
        if not options:
            continue
        best_angle = 180.0
        for opt in options:
            f = opt.transform.get_forward_vector()
            ang = angle_between_deg(ego_dir, vec2(f.x, f.y))
            if ang < best_angle:
                best_angle = ang

        # Prefer setups with a clear straight continuation
        if best_angle > 25.0:
            continue

        score = (len(pairs), abs(d_to_j - 40.0), best_angle)
        if best is None or score < best[0]:
            best = (score, i, sp, wp0, j_wp, junc)

    if best is None:
        # fallback: any spawn point
        sp = spawn_points[0]
        wp0 = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        j_wp, _ = advance_to_junction(wp0, max_dist=140.0, step=7.0)
        junc = j_wp.get_junction() if j_wp else None
        return 0, sp, wp0, j_wp, junc

    _, i, sp, wp0, j_wp, junc = best
    return i, sp, wp0, j_wp, junc


def pick_side_road_incoming_lane(junc: carla.Junction, ego_approach_wp: carla.Waypoint):
    pairs = junc.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return None

    ego_fwd = ego_approach_wp.transform.get_forward_vector()
    ego_dir = vec2(ego_fwd.x, ego_fwd.y)
    ego_dir, _ = norm2(ego_dir)

    ego_road = ego_approach_wp.road_id
    ego_lane = ego_approach_wp.lane_id

    best = None
    for w_in, w_out in pairs:
        if w_in.road_id == ego_road and w_in.lane_id == ego_lane:
            continue

        in_fwd = w_in.transform.get_forward_vector()
        in_dir = vec2(in_fwd.x, in_fwd.y)
        in_dir, _ = norm2(in_dir)

        perp = abs(float(np.dot(in_dir, ego_dir)))  # near 0 is perpendicular
        if perp > 0.55:
            continue

        # Prefer lanes whose junction entry is near the ego junction waypoint location (a true meeting point)
        d = w_in.transform.location.distance(ego_approach_wp.transform.location)
        score = perp + 0.02 * d

        if best is None or score < best[0]:
            best = (score, w_in)

    return best[1] if best else None


def ensure_route_file(client: carla.Client, start_loc: carla.Location, end_loc: carla.Location, route_path: str) -> str:
    if os.path.exists(route_path):
        return route_path
    wps = location_to_waypoint(client, start_loc, end_loc)
    route_maker(wps, route_path)
    return route_path


def draw_debug_stop_line(world: carla.World, wp: carla.Waypoint, life=15.0):
    # Draw an approximate "stop line" across the lane, slightly before junction entry.
    # This is purely visual to help viewers; CARLA stop lines are map-dependent.
    dbg = world.debug
    tf = wp.transform
    loc = tf.location
    right = tf.get_right_vector()
    lane_w = float(max(3.2, min(5.0, wp.lane_width)))
    a = loc + carla.Location(x=right.x * (lane_w * 0.65), y=right.y * (lane_w * 0.65), z=0.15)
    b = loc - carla.Location(x=right.x * (lane_w * 0.65), y=right.y * (lane_w * 0.65), z=-0.15)
    dbg.draw_line(a, b, thickness=0.12, color=carla.Color(255, 255, 255), life_time=life)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = setup_world(client)
    cmap = world.get_map()
    blueprints = world.get_blueprint_library()

    ego = None
    attacker = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()
    original_settings = world.get_settings()

    try:
        # --------- Select T-intersection-like setup ----------
        sp_idx, ego_spawn, ego_wp0, ego_j_wp, junc = find_t_intersection_ego_spawn(world)

        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # --------- Make a "straight through" route for PCLA ----------
        # Choose an endpoint by walking along the ego lane well past the junction.
        # This encourages going straight and not turning at the junction.
        wp_cursor = cmap.get_waypoint(ego_spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        end_wp = wp_cursor
        for _ in range(25):
            nxts = end_wp.next(12.0)
            if not nxts:
                break
            if len(nxts) == 1:
                end_wp = nxts[0]
            else:
                # choose the most straight option relative to current heading
                cur_fwd = end_wp.transform.get_forward_vector()
                cur_dir = vec2(cur_fwd.x, cur_fwd.y)
                cur_dir, _ = norm2(cur_dir)
                best_opt = nxts[0]
                best_ang = 999.0
                for opt in nxts:
                    f = opt.transform.get_forward_vector()
                    ang = angle_between_deg(cur_dir, vec2(f.x, f.y))
                    if ang < best_ang:
                        best_ang = ang
                        best_opt = opt
                end_wp = best_opt

        route_path = "./_pcla_route.xml"
        ensure_route_file(client, ego_spawn.location, end_wp.transform.location, route_path)

        pcla = PCLA("carl_carlv11", ego, route_path, client)

        # --------- Spectator + Camera (attached to spectator) ----------
        spectator = world.get_spectator()

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            image_queue.put(arr[:, :, :3])

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # --------- Attacker: side road vehicle that creeps past stop line ----------
        attacker_bp = blueprints.filter("vehicle.audi.*")[0]

        attacker_entry_wp = None
        attacker_stop_wp = None

        if ego_j_wp is not None and junc is not None:
            # Choose a side-road incoming lane inside the junction boundary
            attacker_entry_wp = pick_side_road_incoming_lane(junc, ego_wp0)

        if attacker_entry_wp is not None:
            # Place attacker just BEFORE it enters the junction boundary (approx behind "stop line")
            prevs = attacker_entry_wp.previous(8.0)
            attacker_stop_wp = prevs[0] if prevs else attacker_entry_wp

            spawn_tf = attacker_stop_wp.transform
            spawn_tf.location.z += 0.35
            attacker = world.try_spawn_actor(attacker_bp, spawn_tf)
        else:
            # Fallback: spawn near ego junction, on the right side, facing toward ego lane
            ego_tf = ego.get_transform()
            fwd = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()
            spawn_loc = ego_tf.location + fwd * 35.0 + right * 10.0
            spawn_loc.z += 0.35
            spawn_rot = carla.Rotation(yaw=ego_tf.rotation.yaw - 90.0)
            attacker = world.try_spawn_actor(attacker_bp, carla.Transform(spawn_loc, spawn_rot))

        if attacker is not None:
            attacker.set_autopilot(False)
            attacker.set_simulate_physics(True)
            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        # Visualize approximate stop line at attacker stop position
        if attacker_stop_wp is not None:
            draw_debug_stop_line(world, attacker_stop_wp, life=DURATION_SECONDS)

        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        attacker_started = False
        attacker_finished = False

        # Precompute attacker intrusion target: a point slightly inside the junction, toward ego lane centerline
        intrusion_target = None
        if attacker_entry_wp is not None and ego_j_wp is not None:
            # Move toward ego_j_wp, but not all the way; just enough to be "in/near ego lane"
            f = attacker_entry_wp.transform.get_forward_vector()
            entry_fwd = vec2(f.x, f.y)
            entry_fwd, _ = norm2(entry_fwd)

            # Use ego junction waypoint location as "conflict zone"
            tgt = ego_j_wp.transform.location
            intrusion_target = carla.Location(tgt.x, tgt.y, tgt.z)

        # --------- Main loop (15 seconds) ----------
        while True:
            snap = world.get_snapshot()
            now = snap.timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # Ego control via PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation

            # Distance to junction for timing
            ego_to_j = None
            if ego_j_wp is not None:
                ego_to_j = ego_loc.distance(ego_j_wp.transform.location)

            # Start attacker slow roll as ego approaches the intersection
            if attacker is not None and not attacker_started:
                trigger = False
                if ego_to_j is not None:
                    trigger = ego_to_j < TRIGGER_EGO_TO_JUNCTION
                else:
                    trigger = (now - sim_start) > 4.0

                if trigger:
                    attacker_started = True
                    attacker.apply_control(carla.VehicleControl(throttle=0.22, brake=0.0))
                    print("[EVENT] Attacker begins rolling past stop line")

            # Drive attacker forward a small amount into ego lane / conflict area, then stop
            if attacker is not None and attacker_started and not attacker_finished:
                a_loc = attacker.get_location()

                if intrusion_target is not None:
                    v = vec2(intrusion_target.x - a_loc.x, intrusion_target.y - a_loc.y)
                    v_dir, v_norm = norm2(v)

                    # Stop once attacker has intruded enough (close to conflict zone)
                    if v_norm < 5.0:
                        attacker_finished = True
                        if ATTACKER_STOP_AFTER_INTRUDE:
                            attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        print("[EVENT] Attacker intruded into/near ego lane and stopped")
                    else:
                        attacker.set_target_velocity(
                            carla.Vector3D(float(v_dir[0] * ATTACKER_SPEED), float(v_dir[1] * ATTACKER_SPEED), 0.0)
                        )
                else:
                    # Fallback: roll forward in its current direction for a short distance, then stop
                    if attacker_stop_wp is not None:
                        # approximate stop line: once moved forward ATTACKER_INTRUDE_DEPTH beyond it, stop
                        stop_line_loc = attacker_stop_wp.transform.location
                        if a_loc.distance(stop_line_loc) > (8.0 + ATTACKER_INTRUDE_DEPTH):
                            attacker_finished = True
                            attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            print("[EVENT] Attacker rolled past stop line and stopped (fallback)")
                    else:
                        # last resort
                        attacker_finished = True

            # Chase camera: slightly above and behind ego, aligned with ego's perspective
            fwd = ego_tf.get_forward_vector()
            cam_loc = ego_loc - fwd * 7.5 + carla.Location(z=3.2)
            cam_rot = carla.Rotation(pitch=-10.0, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # Tick and record
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

        try:
            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
