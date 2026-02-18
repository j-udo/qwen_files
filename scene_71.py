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

# Desired ego setup: through road approaching a T-intersection (not a roundabout)
EGO_TO_JUNCTION_MIN = 35.0
EGO_TO_JUNCTION_MAX = 75.0

# Attacker: comes from side road, rolls past stop line into ego lane vicinity
TRIGGER_EGO_TO_JUNCTION = 22.0   # start rolling when ego is close to junction
ATTACKER_SPEED = 2.2            # m/s (slow roll)
ATTACKER_MAX_ROLL_TIME = 6.0    # seconds, safety bound
ATTACKER_STOP_AFTER_INTRUDE = True

# How deep into the conflict zone we want the attacker to creep
INTRUSION_TARGET_DISTANCE_TO_EGO_CENTER = 1.6  # meters (close to ego lane center)
INTRUSION_TARGET_MAX_ADVANCE = 10.0            # meters from stop position (avoid fully crossing)

CAM_BACK = 7.5
CAM_UP = 3.2
CAM_PITCH = -10.0


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def v2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def norm(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return v, 0.0
    return v / n, n


def angle_deg(a, b):
    a, _ = norm(a)
    b, _ = norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def advance_to_junction(wp: carla.Waypoint, max_dist=160.0, step=6.0):
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


def is_roundabout_like(junc: carla.Junction) -> bool:
    pairs = junc.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return True
    # Roundabouts tend to have lots of internal lane segments.
    if len(pairs) > 28:
        return True
    # Also reject very "loop-ish" by checking diversity of headings
    yaws = []
    for w_in, w_out in pairs[: min(len(pairs), 20)]:
        yaws.append(w_in.transform.rotation.yaw)
    if len(yaws) >= 8:
        # if headings cover many directions, might be roundabout/complex
        yaws = np.array(yaws, dtype=np.float32)
        spread = float(np.std(np.unwrap(np.deg2rad(yaws))))
        if spread > 0.9:
            return True
    return False


def find_t_intersection_setup(world: carla.World):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    best = None
    for i, sp in enumerate(spawn_points):
        wp0 = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            continue

        j_wp, d_to_j = advance_to_junction(wp0, max_dist=180.0, step=7.0)
        if j_wp is None:
            continue
        if not (EGO_TO_JUNCTION_MIN <= d_to_j <= EGO_TO_JUNCTION_MAX):
            continue

        junc = j_wp.get_junction()
        if junc is None:
            continue
        if is_roundabout_like(junc):
            continue

        pairs = junc.get_waypoints(carla.LaneType.Driving)
        if not pairs:
            continue

        # Ego should go "straight-ish" through the junction (through road)
        ego_fwd = wp0.transform.get_forward_vector()
        ego_dir = v2(ego_fwd.x, ego_fwd.y)
        ego_dir, _ = norm(ego_dir)

        # Get a waypoint just before junction
        wp_before = wp0
        for _ in range(12):
            nxt = wp_before.next(6.0)
            if not nxt:
                break
            if nxt[0].is_junction:
                break
            wp_before = nxt[0]

        options = wp_before.next(18.0)
        if not options:
            continue

        best_ang = 999.0
        for opt in options:
            f = opt.transform.get_forward_vector()
            ang = angle_deg(ego_dir, v2(f.x, f.y))
            best_ang = min(best_ang, ang)

        if best_ang > 18.0:
            continue

        # Heuristic for T: junction should have at least one perpendicular approach,
        # and fewer total connections than a big 4-way.
        # We check for existence of an incoming lane whose direction is near perpendicular to ego.
        perp_found = False
        for w_in, w_out in pairs:
            f = w_in.transform.get_forward_vector()
            in_dir = v2(f.x, f.y)
            in_dir, _ = norm(in_dir)
            dp = abs(float(np.dot(in_dir, ego_dir)))
            if dp < 0.35:  # near perpendicular
                perp_found = True
                break
        if not perp_found:
            continue

        # Prefer simpler junctions (often T) but allow small 4-ways.
        complexity = len(pairs)
        if complexity > 22:
            continue

        score = (complexity, abs(d_to_j - 55.0), best_ang)
        if best is None or score < best[0]:
            best = (score, i, sp, wp0, j_wp, junc)

    if best is None:
        # fallback: pick a mid spawn; scenario may not be perfect but will run
        i = min(31, len(spawn_points) - 1)
        sp = spawn_points[i]
        wp0 = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        j_wp, _ = advance_to_junction(wp0, max_dist=180.0, step=7.0) if wp0 else (None, None)
        junc = j_wp.get_junction() if j_wp else None
        return i, sp, wp0, j_wp, junc

    _, i, sp, wp0, j_wp, junc = best
    return i, sp, wp0, j_wp, junc


def pick_side_road_incoming_lane(junc: carla.Junction, ego_approach_wp: carla.Waypoint):
    pairs = junc.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return None, None

    ego_fwd = ego_approach_wp.transform.get_forward_vector()
    ego_dir = v2(ego_fwd.x, ego_fwd.y)
    ego_dir, _ = norm(ego_dir)

    ego_road = ego_approach_wp.road_id

    best = None
    for w_in, w_out in pairs:
        # must be a different road than ego approach road (a side road)
        if w_in.road_id == ego_road:
            continue

        f = w_in.transform.get_forward_vector()
        in_dir = v2(f.x, f.y)
        in_dir, _ = norm(in_dir)

        # Prefer close-to-perpendicular incoming lanes
        dp = abs(float(np.dot(in_dir, ego_dir)))
        if dp > 0.4:
            continue

        # Prefer lanes whose "out" heads into ego road (conflict)
        # We don't have road option, so just prefer those near junction center
        jc = junc.bounding_box.location
        d_center = w_in.transform.location.distance(jc)

        score = (dp, d_center)
        if best is None or score < best[0]:
            best = (score, w_in, w_out)

    if best is None:
        return None, None
    return best[1], best[2]


def ensure_route_file(client: carla.Client, start_loc: carla.Location, end_loc: carla.Location, route_path: str) -> str:
    if os.path.exists(route_path):
        return route_path
    wps = location_to_waypoint(client, start_loc, end_loc)
    route_maker(wps, route_path)
    return route_path


def draw_stop_line(world: carla.World, wp: carla.Waypoint, life: float):
    dbg = world.debug
    tf = wp.transform
    loc = tf.location
    right = tf.get_right_vector()
    lane_w = float(max(3.2, min(5.0, wp.lane_width)))
    a = loc + carla.Location(x=right.x * (lane_w * 0.7), y=right.y * (lane_w * 0.7), z=0.12)
    b = loc - carla.Location(x=right.x * (lane_w * 0.7), y=right.y * (lane_w * 0.7), z=0.12)
    dbg.draw_line(a, b, thickness=0.14, color=carla.Color(255, 255, 255), life_time=life)
    dbg.draw_string(loc + carla.Location(z=0.6), "STOP LINE (approx)", draw_shadow=True,
                    color=carla.Color(255, 255, 255), life_time=life)


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = setup_world(client)
    cmap = world.get_map()
    blueprints = world.get_blueprint_library()

    original_settings = world.get_settings()

    ego = None
    attacker = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    try:
        sp_idx, ego_spawn, ego_wp0, ego_j_wp, junc = find_t_intersection_setup(world)
        print(f"[INFO] Using spawn point idx={sp_idx}")

        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # Build a route that continues straight through the junction:
        # walk forward and always pick the most straight option.
        wp_cursor = cmap.get_waypoint(ego_spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        end_wp = wp_cursor
        for _ in range(28):
            nxts = end_wp.next(12.0) if end_wp else []
            if not nxts:
                break
            if len(nxts) == 1:
                end_wp = nxts[0]
            else:
                cur_fwd = end_wp.transform.get_forward_vector()
                cur_dir = v2(cur_fwd.x, cur_fwd.y)
                cur_dir, _ = norm(cur_dir)
                best_opt = nxts[0]
                best_ang = 999.0
                for opt in nxts:
                    f = opt.transform.get_forward_vector()
                    ang = angle_deg(cur_dir, v2(f.x, f.y))
                    if ang < best_ang:
                        best_ang = ang
                        best_opt = opt
                end_wp = best_opt

        route_path = "./_pcla_route.xml"
        ensure_route_file(client, ego_spawn.location, end_wp.transform.location, route_path)
        pcla = PCLA("carl_carlv11", ego, route_path, client)

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

        # ---- Attacker setup: pick side road incoming, stop before junction (stop line), then creep forward ----
        attacker_bp = blueprints.filter("vehicle.audi.*")[0]

        attacker_in_wp = None
        attacker_out_wp = None
        attacker_stop_wp = None

        if ego_j_wp is not None and junc is not None and ego_wp0 is not None:
            attacker_in_wp, attacker_out_wp = pick_side_road_incoming_lane(junc, ego_wp0)

        if attacker_in_wp is not None:
            prevs = attacker_in_wp.previous(10.0)
            attacker_stop_wp = prevs[0] if prevs else attacker_in_wp

            spawn_tf = attacker_stop_wp.transform
            spawn_tf.location.z += 0.35
            attacker = world.try_spawn_actor(attacker_bp, spawn_tf)
        else:
            # Fallback: spawn on a perpendicular offset ahead of ego, oriented to cross into ego lane
            ego_tf0 = ego.get_transform()
            fwd = ego_tf0.get_forward_vector()
            right = ego_tf0.get_right_vector()
            spawn_loc = ego_tf0.location + fwd * 45.0 + right * 14.0
            spawn_loc.z += 0.35
            spawn_rot = carla.Rotation(yaw=ego_tf0.rotation.yaw - 90.0)
            attacker = world.try_spawn_actor(attacker_bp, carla.Transform(spawn_loc, spawn_rot))

        if attacker is not None:
            attacker.set_autopilot(False)
            attacker.set_simulate_physics(True)
            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        if attacker_stop_wp is not None:
            draw_stop_line(world, attacker_stop_wp, life=DURATION_SECONDS)

        # Compute an intrusion target near ego lane center at the conflict zone (junction entry)
        intrusion_target = None
        if attacker is not None and attacker_stop_wp is not None and ego_j_wp is not None:
            # Use ego junction waypoint as conflict point, but aim slightly into ego lane centerline
            conflict = ego_j_wp.transform.location
            intrusion_target = carla.Location(conflict.x, conflict.y, conflict.z)

        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        attacker_started = False
        attacker_done = False
        attacker_start_time = None
        attacker_stop_loc = attacker.get_location() if attacker is not None else None

        while True:
            snap = world.get_snapshot()
            now = snap.timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # Ego control
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation

            # Distance to junction for timing
            ego_to_j = None
            if ego_j_wp is not None:
                ego_to_j = ego_loc.distance(ego_j_wp.transform.location)

            # Start attacker roll as ego approaches
            if attacker is not None and (not attacker_started):
                trigger = False
                if ego_to_j is not None:
                    trigger = ego_to_j < TRIGGER_EGO_TO_JUNCTION
                else:
                    trigger = (now - sim_start) > 5.0

                if trigger:
                    attacker_started = True
                    attacker_start_time = now
                    attacker_stop_loc = attacker.get_location()
                    attacker.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0))
                    print("[EVENT] Attacker begins rolling past stop line")

            # Attacker motion: creep toward intrusion target (or just forward) but limit travel/time
            if attacker is not None and attacker_started and (not attacker_done):
                a_loc = attacker.get_location()

                # Safety timeout
                if attacker_start_time is not None and (now - attacker_start_time) > ATTACKER_MAX_ROLL_TIME:
                    attacker_done = True
                    attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    print("[EVENT] Attacker stopped (timeout safety)")

                if not attacker_done:
                    # Limit how far it advances from its stop position
                    if attacker_stop_loc is not None and a_loc.distance(attacker_stop_loc) > INTRUSION_TARGET_MAX_ADVANCE:
                        attacker_done = True
                        attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                        attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        print("[EVENT] Attacker stopped (max advance safety)")

                if not attacker_done:
                    if intrusion_target is not None:
                        # Aim toward a point near ego lane center: if close enough, stop while intruding
                        v = v2(intrusion_target.x - a_loc.x, intrusion_target.y - a_loc.y)
                        v_dir, v_norm = norm(v)

                        if v_norm < INTRUSION_TARGET_DISTANCE_TO_EGO_CENTER:
                            attacker_done = True
                            if ATTACKER_STOP_AFTER_INTRUDE:
                                attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                                attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            print("[EVENT] Attacker intruded into/near ego lane and stopped")
                        else:
                            attacker.set_target_velocity(
                                carla.Vector3D(float(v_dir[0] * ATTACKER_SPEED), float(v_dir[1] * ATTACKER_SPEED), 0.0)
                            )
                    else:
                        # Fallback: move along its current forward direction slowly
                        tf = attacker.get_transform()
                        f = tf.get_forward_vector()
                        attacker.set_target_velocity(carla.Vector3D(f.x * ATTACKER_SPEED, f.y * ATTACKER_SPEED, 0.0))

            # Camera: slightly above and behind ego, aligned with ego yaw, forward-facing
            fwd = ego_tf.get_forward_vector()
            cam_loc = ego_loc - fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

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
