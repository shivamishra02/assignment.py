# Build a ready-to-run project scaffold with code, config, and docs.
import os, json, textwrap, datetime, pathlib
root = "/mnt/data/athlete_rise_cover_drive"
output_dir = os.path.join(root, "output")
os.makedirs(output_dir, exist_ok=True)

main_code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AthleteRise – AI-Powered Cricket Analytics
Real-Time Cover Drive Analysis from Full Video

Main script: cover_drive_analysis_realtime.py
Author: ChatGPT (GPT-5 Thinking)

What this does (Base + several bonuses):
- Reads a full video (local path OR YouTube URL) and processes frames sequentially.
- Runs MediaPipe Pose on each frame.
- Computes rolling biomechanics:
    * Front elbow angle (shoulder–elbow–wrist)
    * Spine lean (hip–shoulder line vs. vertical)
    * Head-over-knee alignment (x-distance, normalized by width)
    * Front foot direction (heel→toe angle vs. x-axis, a crease surrogate)
- Draws live overlays on the video (skeleton, metrics, and feedback cues).
- Writes one annotated video in ./output/annotated_video.mp4.
- Generates final multi-category evaluation with scores (1–10) + feedback in ./output/evaluation.json
- Handles missing detections gracefully.
- Logs average FPS.
- (Bonus) Simple phase segmentation + contact-heuristic
- (Bonus) Smoothing + charts saved to ./output/metrics_chart.png
- (Bonus) Reference comparison + skill grade + tiny HTML report

Usage (examples):
    python cover_drive_analysis_realtime.py --source "https://youtube.com/shorts/vSX3IRxGnNY"
    python cover_drive_analysis_realtime.py --source /path/to/video.mp4

Notes:
- Internet is required at runtime only if you pass a YouTube URL (uses yt-dlp).
- For speed on CPU, you may set --resize-width 720 and --model-complexity 0.
"""

import os, sys, cv2, time, math, json, argparse, pathlib, statistics
import numpy as np

# ----------------------- YouTube download (optional) -----------------------
def download_youtube(url: str, temp_dir: str) -> str:
    """Download a YouTube video to temp_dir using yt_dlp, fallback to pytube."""
    os.makedirs(temp_dir, exist_ok=True)
    target = os.path.join(temp_dir, "input_youtube.mp4")
    try:
        import yt_dlp
        ydl_opts = {
            "outtmpl": target,
            "format": "mp4/bestaudio/best",
            "merge_output_format": "mp4",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if os.path.exists(target):
            return target
    except Exception as e:
        print("[WARN] yt_dlp failed:", e, file=sys.stderr)
    try:
        from pytube import YouTube
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
        stream.download(output_path=temp_dir, filename="input_youtube.mp4")
        return target
    except Exception as e:
        print("[ERROR] pytube failed:", e, file=sys.stderr)
        raise RuntimeError("Could not download YouTube video") from e

# ----------------------- Geometry helpers -----------------------
def angle(a, b, c):
    """Angle ABC in degrees given points (x, y)."""
    if a is None or b is None or c is None:
        return None
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cosang = max(-1.0, min(1.0, dot/(mag_ba*mag_bc)))
    return math.degrees(math.acos(cosang))

def angle_with_xaxis(vx, vy):
    """Unsigned angle in degrees between vector and +x axis."""
    try:
        return abs(math.degrees(math.atan2(vy, vx)))
    except Exception:
        return None

def ema(prev, new, alpha=0.3):
    if new is None: return prev
    if prev is None: return new
    return alpha*new + (1-alpha)*prev

# ----------------------- MediaPipe setup -----------------------
def init_pose(static=False, model_complexity=1):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp, mp_pose, pose

def extract_landmarks(results, w, h):
    """Return dict of pixels for key joints or None if not visible enough."""
    names = [
        "NOSE",
        "LEFT_SHOULDER","RIGHT_SHOULDER",
        "LEFT_ELBOW","RIGHT_ELBOW",
        "LEFT_WRIST","RIGHT_WRIST",
        "LEFT_HIP","RIGHT_HIP",
        "LEFT_KNEE","RIGHT_KNEE",
        "LEFT_ANKLE","RIGHT_ANKLE",
        "LEFT_HEEL","RIGHT_HEEL",
        "LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
    ]
    pts = {n: None for n in names}
    if not results or not results.pose_landmarks:
        return pts
    lm = results.pose_landmarks.landmark
    from mediapipe.solutions.pose import PoseLandmark as L
    for n in names:
        try:
            p = lm[getattr(L, n).value]
            if p.visibility is not None and p.visibility < 0.3:
                pts[n] = None
            else:
                pts[n] = (p.x*w, p.y*h)
        except Exception:
            pts[n] = None
    return pts

# ----------------------- Metrics -----------------------
def front_side_from_feet(pts):
    """Heuristic: which ankle extends further in x from mid-hip."""
    lh, rh = pts.get("LEFT_HIP"), pts.get("RIGHT_HIP")
    la, ra = pts.get("LEFT_ANKLE"), pts.get("RIGHT_ANKLE")
    if lh and rh and la and ra:
        midx = (lh[0]+rh[0])/2.0
        return "LEFT" if abs(la[0]-midx) > abs(ra[0]-midx) else "RIGHT"
    if la and ra:
        return "LEFT" if la[1] > ra[1] else "RIGHT"
    return None

def compute_metrics(pts, w, h, ema_state, alpha=0.3):
    front = front_side_from_feet(pts)
    # mid points
    ls, rs = pts.get("LEFT_SHOULDER"), pts.get("RIGHT_SHOULDER")
    lh, rh = pts.get("LEFT_HIP"), pts.get("RIGHT_HIP")
    shoulders_mid = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0) if (ls and rs) else None
    hips_mid = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0) if (lh and rh) else None

    # front elbow
    if front == "LEFT":
        sh, el, wr = pts.get("LEFT_SHOULDER"), pts.get("LEFT_ELBOW"), pts.get("LEFT_WRIST")
    else:
        sh, el, wr = pts.get("RIGHT_SHOULDER"), pts.get("RIGHT_ELBOW"), pts.get("RIGHT_WRIST")
    elbow_angle = angle(sh, el, wr)

    # spine lean vs vertical (0=vertical)
    spine_lean = None
    if shoulders_mid and hips_mid:
        vx, vy = shoulders_mid[0]-hips_mid[0], shoulders_mid[1]-hips_mid[1]
        # angle to vertical up vector (0,-1)
        dot = vx*0 + vy*(-1)
        mag = math.hypot(vx, vy)
        if mag > 0:
            cosang = max(-1.0, min(1.0, dot/(mag*1.0)))
            spine_lean = abs(90 - abs(math.degrees(math.acos(cosang))))

    # head over front knee (x distance normalized by width)
    head = pts.get("NOSE")
    fk = pts.get(f"{front}_KNEE") if front else None
    head_knee_dx_norm = None
    if head and fk and w>0:
        head_knee_dx_norm = abs(head[0]-fk[0])/float(w)

    # front foot direction: heel->toe vs x-axis
    heel = pts.get(f"{front}_HEEL") if front else None
    toe = pts.get(f"{front}_FOOT_INDEX") if front else None
    foot_dir = angle_with_xaxis(toe[0]-heel[0], toe[1]-heel[1]) if (heel and toe) else None

    # smooth
    ema_state["elbow"] = ema(ema_state.get("elbow"), elbow_angle, alpha) if elbow_angle is not None else ema_state.get("elbow")
    ema_state["spine"] = ema(ema_state.get("spine"), spine_lean, alpha) if spine_lean is not None else ema_state.get("spine")
    ema_state["headknee"] = ema(ema_state.get("headknee"), head_knee_dx_norm, alpha) if head_knee_dx_norm is not None else ema_state.get("headknee")
    ema_state["footdir"] = ema(ema_state.get("footdir"), foot_dir, alpha) if foot_dir is not None else ema_state.get("footdir")

    return {
        "front_side": front,
        "elbow_angle_front": ema_state.get("elbow"),
        "spine_lean_deg": ema_state.get("spine"),
        "head_over_knee_dx_norm": ema_state.get("headknee"),
        "front_foot_dir_deg": ema_state.get("footdir"),
    }, ema_state

# ----------------------- Thresholds & Scoring -----------------------
DEFAULT_THRESHOLDS = dict(
    elbow_min=100.0, elbow_max=160.0,
    spine_lean_max=20.0,
    head_knee_dx_norm_max=0.06,
    foot_dir_deg_max=30.0
)

REFERENCE_RANGES = dict(
    elbow=(110.0,150.0),
    spine_lean=(0.0,15.0),
    head_knee_dx_norm=(0.0,0.05),
    foot_dir_deg=(0.0,25.0),
)

def live_feedback(metrics, th=DEFAULT_THRESHOLDS):
    cues = []
    e = metrics.get("elbow_angle_front")
    if e is not None:
        cues.append("✅ Good elbow elevation" if th["elbow_min"] <= e <= th["elbow_max"] else "❌ Adjust elbow elevation")
    s = metrics.get("spine_lean_deg")
    if s is not None:
        cues.append("✅ Balanced spine" if s <= th["spine_lean_max"] else "❌ Excessive spine lean")
    hk = metrics.get("head_over_knee_dx_norm")
    if hk is not None:
        cues.append("✅ Head over front knee" if hk <= th["head_knee_dx_norm_max"] else "❌ Head not over front knee")
    fd = metrics.get("front_foot_dir_deg")
    if fd is not None:
        cues.append("✅ Front foot aligned" if fd <= th["foot_dir_deg_max"] else "❌ Front foot splayed")
    return cues

def score_and_feedback(metric_tracks):
    def score_range(values, lo, hi):
        if not values: return 5.0
        avg = float(np.mean(values))
        if lo <= avg <= hi: return 9.0
        d = min(abs(avg-lo), abs(avg-hi))
        span = max(hi-lo, 1e-6)
        return max(1.0, 9.0 - 8.0*(d/span))

    e = metric_tracks.get("elbow_angle_front", [])
    s = metric_tracks.get("spine_lean_deg", [])
    hk = metric_tracks.get("head_over_knee_dx_norm", [])
    fd = metric_tracks.get("front_foot_dir_deg", [])

    scores = {}
    scores["Footwork"] = {
        "score": round(score_range(fd, *REFERENCE_RANGES["foot_dir_deg"]), 1),
        "comment": "Keep the front foot pointing along the crease; reduce splay in the stride."
    }
    scores["Head Position"] = {
        "score": round(score_range(hk, *REFERENCE_RANGES["head_knee_dx_norm"]), 1),
        "comment": "Keep head stacked over front knee at impact for stability."
    }
    scores["Swing Control"] = {
        "score": round(score_range(e, *REFERENCE_RANGES["elbow"]), 1),
        "comment": "Maintain a firm, elevated front elbow to guide a straight bat swing."
    }
    scores["Balance"] = {
        "score": round(score_range(s, *REFERENCE_RANGES["spine_lean"]), 1),
        "comment": "Limit lateral spine lean; transfer weight forward without collapsing."
    }
    scores["Follow-through"] = {
        "score": round(np.mean([scores["Swing Control"]["score"], scores["Balance"]["score"]]), 1),
        "comment": "Allow a relaxed follow-through while keeping a stable base."
    }
    overall = float(np.mean([v["score"] for v in scores.values()]))
    scores["_overall"] = {"average": round(overall,1), "grade": "Advanced" if overall>=8 else ("Intermediate" if overall>=5.5 else "Beginner")}
    return scores

# ----------------------- Phases & Contact (simple heuristics) -----------------------
class PhaseTracker:
    STATES = ["Stance", "Stride", "Downswing", "Impact", "Follow-through", "Recovery"]
    def __init__(self):
        self.state = "Stance"
        self.prev_pts = None
        self.impact_index = None

    def _vel(self, a, b):
        if a is None or b is None: return 0.0
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, idx, pts):
        # very lightweight: use wrist velocity peaks for downswing/impact hints
        lw, rw = pts.get("LEFT_WRIST"), pts.get("RIGHT_WRIST")
        lv = rv = 0.0
        if self.prev_pts is not None:
            lv = self._vel(lw, self.prev_pts.get("LEFT_WRIST"))
            rv = self._vel(rw, self.prev_pts.get("RIGHT_WRIST"))
        wrist_v = max(lv, rv)

        # state transitions (heuristic)
        if self.state == "Stance" and wrist_v > 4:
            self.state = "Stride"
        elif self.state == "Stride" and wrist_v > 8:
            self.state = "Downswing"
        elif self.state == "Downswing" and wrist_v > 14:
            self.state = "Impact"
            self.impact_index = idx
        elif self.state == "Impact" and wrist_v < 6:
            self.state = "Follow-through"
        elif self.state == "Follow-through" and wrist_v < 2:
            self.state = "Recovery"

        self.prev_pts = pts
        return self.state

# ----------------------- Drawing helpers -----------------------
def draw_pose(frame, mp, mp_pose, results, color=(0,255,0)):
    if results and results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

def put_text(frame, text, org, scale=0.6, thickness=2, color=(255,255,255), bg=True):
    if bg:
        (w,h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (org[0]-3, org[1]-h-3), (org[0]+w+3, org[1]+3), (0,0,0), -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ----------------------- Analyze video -----------------------
def analyze_video(source:str, out_dir:str="./output", resize_width:int=960, model_complexity:int=1, ema_alpha:float=0.3, debug_contact:bool=False):
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Resolve source (YouTube or file path)
    if source.startswith("http://") or source.startswith("https://"):
        vid_path = download_youtube(source, tmp_dir)
    else:
        vid_path = source
    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"Video not found: {vid_path}")

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1.0
    target_w = resize_width if resize_width>0 else w
    if w>0: scale = target_w / float(w)
    out_w = int(w*scale)
    out_h = int(h*scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # Pose
    mp, mp_pose, pose = init_pose(static=False, model_complexity=model_complexity)

    # Tracking
    ema_state = {"elbow":None,"spine":None,"headknee":None,"footdir":None}
    metric_tracks = {"elbow_angle_front":[], "spine_lean_deg":[], "head_over_knee_dx_norm":[], "front_foot_dir_deg":[]}
    phase = PhaseTracker()

    # Loop
    t0 = time.time()
    frame_idx = 0
    contact_frames = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pts = extract_landmarks(results, out_w, out_h)

            # metrics
            m, ema_state = compute_metrics(pts, out_w, out_h, ema_state, alpha=ema_alpha)
            for k in metric_tracks.keys():
                if m.get(k) is not None: metric_tracks[k].append(float(m[k]))

            # phase & crude contact detection (wrist velocity spike already updates inside)
            state = phase.update(frame_idx, pts)
            if state == "Impact":
                contact_frames.append(frame_idx)
                if debug_contact:
                    cv2.imwrite(os.path.join(tmp_dir, f"contact_{frame_idx:06d}.jpg"), frame)

            # draw
            draw_pose(frame, mp, mp_pose, results)
            y = 24
            put_text(frame, f"Elbow: {m.get('elbow_angle_front') and round(m['elbow_angle_front'],1)} deg", (10,y)); y+=22
            put_text(frame, f"Spine lean: {m.get('spine_lean_deg') and round(m['spine_lean_deg'],1)} deg", (10,y)); y+=22
            put_text(frame, f"Head-knee dx: {m.get('head_over_knee_dx_norm') and round(m['head_over_knee_dx_norm'],3)}", (10,y)); y+=22
            put_text(frame, f"Front foot dir: {m.get('front_foot_dir_deg') and round(m['front_foot_dir_deg'],1)} deg", (10,y)); y+=26

            cues = live_feedback(m)
            for c in cues[:3]:
                put_text(frame, c, (10,y)); y+=22

            put_text(frame, f"Phase: {state}", (10, out_h-12))

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        try: pose.close()
        except Exception: pass

    t1 = time.time()
    avg_fps = frame_idx / max(1e-6, (t1-t0))
    print(f"[INFO] Processed {frame_idx} frames in {t1-t0:.2f}s -> avg {avg_fps:.2f} FPS")

    # Final scoring
    scores = score_and_feedback(metric_tracks)

    # Save evaluation
    eval_path = os.path.join(out_dir, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({
            "video": os.path.abspath(vid_path),
            "frames": frame_idx,
            "avg_fps": round(avg_fps,2),
            "scores": scores,
            "contact_frames": contact_frames[:3]  # a hint, not guaranteed exact
        }, f, indent=2)

    # Plot charts (elapsed vs elbow/spine) as PNG
    try:
        import matplotlib.pyplot as plt
        t = np.arange(len(metric_tracks["elbow_angle_front"])) / (fps if fps>0 else 30.0)
        plt.figure()
        if metric_tracks["elbow_angle_front"]:
            plt.plot(t, metric_tracks["elbow_angle_front"], label="Elbow (deg)")
        if metric_tracks["spine_lean_deg"]:
            t2 = np.arange(len(metric_tracks["spine_lean_deg"]))/(fps if fps>0 else 30.0)
            plt.plot(t2, metric_tracks["spine_lean_deg"], label="Spine lean (deg)")
        plt.xlabel("Time (s)"); plt.ylabel("Degrees"); plt.legend()
        chart_path = os.path.join(out_dir, "metrics_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print("[WARN] chart plot failed:", e, file=sys.stderr)

    # Tiny HTML report
    try:
        html = ["<html><head><meta charset='utf-8'><title>AthleteRise Report</title></head><body>",
                "<h2>AthleteRise – Cover Drive Analysis</h2>",
                f"<p><b>Frames:</b> {frame_idx} | <b>Avg FPS:</b> {avg_fps:.2f}</p>",
                "<h3>Scores</h3><ul>"]
        for k, v in scores.items():
            if k == "_overall": continue
            html.append(f"<li><b>{k}:</b> {v['score']}/10 – {v['comment']}</li>")
        html.append("</ul>")
        html.append(f"<p><b>Overall:</b> {scores['_overall']['average']}/10 – <b>{scores['_overall']['grade']}</b></p>")
        if os.path.exists(os.path.join(out_dir, 'metrics_chart.png')):
            html.append("<h3>Metric Trends</h3><img src='metrics_chart.png' width='600'/>")
        html.append("</body></html>")
        with open(os.path.join(out_dir,"report.html"), "w", encoding="utf-8") as f:
            f.write("\n".join(html))
    except Exception as e:
        print("[WARN] report generation failed:", e, file=sys.stderr)

    return {"out_video": out_path, "evaluation": eval_path}

# ----------------------- CLI -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="AthleteRise – Real-Time Cover Drive Analysis")
    ap.add_argument("--source", required=True, help="Local video path or YouTube URL")
    ap.add_argument("--resize-width", type=int, default=960, help="Resize width (0 = no resize)")
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0,1,2], help="MediaPipe model complexity")
    ap.add_argument("--ema-alpha", type=float, default=0.3, help="EMA smoothing factor (0..1)")
    ap.add_argument("--debug-contact", action="store_true", help="Save contact debug frames")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs("./output", exist_ok=True)
    results = analyze_video(
        source=args.source,
        out_dir="./output",
        resize_width=args.resize_width,
        model_complexity=args.model_complexity,
        ema_alpha=args.ema_alpha,
        debug_contact=args.debug_contact
    )
    print(json.dumps(results, indent=2))
'''

print("Script started")
# ... your existing code ...
print("Script finished")
