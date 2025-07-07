import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time
from collections import defaultdict
from tqdm import tqdm  # Pasek postępu
import torch

# ========== KONFIGURACJA ==========
CONFIDENCE_THRESHOLD = 0.1
TRACKING_THRESHOLD = 150
ROI_EXPANSION = 2.0
SHOW_SKELETON = False  # Wyłącz podgląd
# ==================================

model = YOLO("yolo11n-pose.pt").to("cuda").half()

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    if keypoints.size == 0 or keypoints.shape[0] < 17:
        return
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start = keypoints[start_idx]
        end = keypoints[end_idx]
        if not (np.isnan(start).any() or np.isnan(end).any()):
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)
    for point in keypoints:
        if not np.isnan(point).any():
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 0, 255), -1)

def get_centroid(keypoints):
    return np.nanmean(keypoints, axis=0) if len(keypoints) else np.array([np.nan, np.nan])

def euclidean(p1, p2):
    if np.isnan(p1).any() or np.isnan(p2).any():
        return float('inf')
    return np.linalg.norm(p1 - p2)

def detect_people(frame, conf_threshold=0.25, imgsz=640):
    results = list(model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False, stream=True))
    if not results:
        return [], []
    boxes = results[0].boxes.xyxy.cpu().numpy()
    keypoints = results[0].keypoints.xy.cpu().numpy()
    return boxes, keypoints

def match_people(current_centroids, previous_centroids, max_distance=TRACKING_THRESHOLD):
    matches = {}
    used = set()
    for i, cur in enumerate(current_centroids):
        min_dist = float('inf')
        best_idx = -1
        for j, prev in enumerate(previous_centroids):
            if j in used:
                continue
            dist = euclidean(cur, prev)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_idx = j
        if best_idx != -1:
            matches[i] = best_idx
            used.add(best_idx)
    return matches

# ==== ŚCIEŻKI ====
video_folder = Path("../../trimmed_films")
output_folder = Path("../pose_outputs_auto")

output_folder.mkdir(exist_ok=True)

video_files = list(video_folder.glob("*.mp4"))

for video_file in video_files[:5]:
    print(f"Przetwarzanie: {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracked_people = defaultdict(list)
    previous_centroids = []
    person_ids = []
    person_id_counter = 0

    frame_count = 0

    pbar = tqdm(total=total_frames, desc=f"Przetwarzanie {video_file.name}", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, keypoints_batch = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=640)

        centroids = [get_centroid(kp) for kp in keypoints_batch]
        matches = match_people(centroids, previous_centroids)

        new_person_ids = [-1] * len(centroids)
        used_prev_ids = set()

        for cur_idx, prev_idx in matches.items():
            for pid, prev_id in enumerate(person_ids):
                if prev_idx == pid and pid not in used_prev_ids:
                    new_person_ids[cur_idx] = pid
                    used_prev_ids.add(pid)
                    break

        for i, pid in enumerate(new_person_ids):
            if pid == -1:
                pid = person_id_counter
                person_id_counter += 1
            if keypoints_batch[i].size > 0 and keypoints_batch[i].shape[0] >= 17:
                tracked_people[pid].append(keypoints_batch[i].flatten().tolist())
            else:
                tracked_people[pid].append([np.nan] * 34)
            new_person_ids[i] = pid

        for pid in tracked_people:
            if len(tracked_people[pid]) < frame_count + 1:
                tracked_people[pid].append([np.nan] * 34)

        previous_centroids = centroids
        person_ids = new_person_ids
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    # Wybór osoby z największą liczbą nie-NaN klatek
    best_id = None
    best_valid = -1
    for pid, keypoints in tracked_people.items():
        valid = sum(1 for kp in keypoints if not np.isnan(kp[0]))
        if valid > best_valid:
            best_valid = valid
            best_id = pid

    if best_id is not None:
        df = pd.DataFrame(tracked_people[best_id])
        output_path = output_folder / f"{video_file.stem}.csv"
        df.to_csv(output_path, index=False)

        detection_rate = 100 * best_valid / frame_count

        stats_path = output_folder / "stats.csv"
        stats_entry = pd.DataFrame([{
            "file": video_file.stem,
            "percent_of_detection": round(detection_rate, 2)
        }])

        if stats_path.exists():
            stats_entry.to_csv(stats_path, mode='a', header=False, index=False)
        else:
            stats_entry.to_csv(stats_path, mode='w', header=True, index=False)

        print(f"Zapisano: {output_path}")
        print(f"Wykryto {best_valid} / {frame_count} klatek ({detection_rate:.2f}%)")
        print(f"Dodano statystyki do: {stats_path}")
    else:
        print("Nie wykryto żadnej osoby przez cały film.")
