import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time
import argparse

# ========== KONFIGURACJA ==========
SHOW_SKELETON = True
TRACKING_THRESHOLD = 150
CONFIDENCE_THRESHOLD = 0.1
ROI_EXPANSION = 2.0
USE_FULL_FRAME_DETECTION = True
REACQUIRE_INTERVAL = 1.0  # Co ile sekund próbować ponownie wykryć osobę
# ==================================

model = YOLO("yolov8m-pose.pt")

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
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))
            cv2.line(frame, start, end, color, thickness)
    
    for point in keypoints:
        if not np.isnan(point).any():
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

def get_centroid(keypoints):
    keypoints = np.array(keypoints)
    return np.nanmean(keypoints, axis=0) if len(keypoints) else np.array([np.nan, np.nan])

def euclidean(p1, p2):
    if np.isnan(p1).any() or np.isnan(p2).any():
        return float('inf')
    return np.linalg.norm(p1 - p2)

def get_roi(bbox, frame_shape, expansion=1.5):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    new_x_min = max(0, int(x_min - width * (expansion - 1) / 2))
    new_y_min = max(0, int(y_min - height * (expansion - 1) / 2))
    new_x_max = min(frame_shape[1], int(x_max + width * (expansion - 1) / 2))
    new_y_max = min(frame_shape[0], int(y_max + height * (expansion - 1) / 2))
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)

def detect_people(frame, conf_threshold=0.25, imgsz=640):
    results = model(frame, conf=conf_threshold, imgsz=imgsz, verbose=False)
    if len(results) == 0:
        return np.array([]), np.array([])
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    keypoints = results[0].keypoints.xy.cpu().numpy()
    
    return boxes, keypoints

# Setup argument parser
parser = argparse.ArgumentParser(description="Process ski videos.")
parser.add_argument(
    "--video_folder",
    type=str,
    default="../git_films",
    help="Path to the folder containing ski videos"
)

args = parser.parse_args()
video_folder = Path(args.video_folder)
video_files = list(video_folder.glob("*.mp4"))

output_folder = Path("../pose_outputs_clicked")
output_folder.mkdir(exist_ok=True)

# Globalne zmienne do obsługi interakcji
click_position = []
selected_bbox = None
confirmed = False
current_frame_idx = 0
pause = False

def on_mouse(event, x, y, flags, param):
    global click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = [(x, y)]

def update_frame_pos(pos):
    global current_frame_idx
    current_frame_idx = pos

for video_file in video_files:
    print(f"Przetwarzanie: {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku: {video_file}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Wideo: {total_frames} klatek, {fps:.2f} FPS, rozdzielczość: {width}x{height}")
    
    window_name = "Wybierz osobe: Przewijaj [<] [>], Spacja: Play/Pause, Kliknij + [C] zatwierdz, [Q] wyjscie"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.createTrackbar('Klatka', window_name, 0, total_frames-1, update_frame_pos)
    
    click_position = []
    selected_bbox = None
    confirmed = False
    pause = True
    
    while not confirmed:
        if not pause:
            current_frame_idx = (current_frame_idx + 1) % total_frames
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.setTrackbarPos('Klatka', window_name, current_frame_idx)
        
        boxes, keypoints_batch = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=1280)
        
        display_frame = frame.copy()
        
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = map(int, box[:4])
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_frame, f"#{i}", (x_min+5, y_min+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        if SHOW_SKELETON:
            for kp in keypoints_batch:
                if kp.size > 0 and kp.shape[0] >= 17:
                    draw_skeleton(display_frame, kp, color=(0, 150, 255), thickness=2)
        
        if selected_bbox is not None:
            x_min, y_min, x_max, y_max = map(int, selected_bbox)
            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            
            roi_coords = get_roi(selected_bbox, frame.shape, ROI_EXPANSION)
            rx_min, ry_min, rx_max, ry_max = map(int, roi_coords)
            cv2.rectangle(display_frame, (rx_min, ry_min), (rx_max, ry_max), (255, 0, 0), 2)
            cv2.putText(display_frame, "ROI", (rx_min+5, ry_min+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        info_y = 30
        cv2.putText(display_frame, "Kliknij osobe i nacisnij [C] by zatwierdzic", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "[Spacja]: Play/Pause | [Q]: Wyjscie | [<] [>]: Przewijanie", 
                   (10, info_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Klatka: {current_frame_idx}/{total_frames-1}", 
                   (10, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"ROI: {ROI_EXPANSION}x | conf: {CONFIDENCE_THRESHOLD}", 
                   (10, info_y+90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            pause = not pause
        elif key == ord('c') and selected_bbox is not None:
            confirmed = True
        elif key == ord('.'):
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
            pause = True
        elif key == ord(','):
            current_frame_idx = max(current_frame_idx - 1, 0)
            pause = True
        
        if click_position and len(boxes) > 0:
            click_x, click_y = click_position[0]
            
            min_dist = float('inf')
            closest_idx = -1
            
            for i, box in enumerate(boxes):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            if closest_idx >= 0:
                selected_bbox = boxes[closest_idx]
                print(f"Wybrano osobę #{closest_idx} na klatce {current_frame_idx}")
    
    cv2.destroyWindow(window_name)
    
    if not confirmed or selected_bbox is None:
        print("Pominięto śledzenie dla tego filmu.")
        cap.release()
        continue

    print(f"Rozpoczynam śledzenie osoby w ROI {ROI_EXPANSION}x...")
    all_keypoints = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    tracking_lost_count = 0
    
    # Stan śledzenia
    tracking_active = True
    last_known_bbox = selected_bbox.copy()
    
    # Czas ostatniej próby ponownego wykrycia
    last_reacquire_attempt = time.time()
    
    if SHOW_SKELETON:
        preview_window = "Podglad sledzenia"
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 800, 600)
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = time.time()
        
        # Próba ponownego wykrycia jeśli śledzenie nieaktywne
        if not tracking_active:
            # Ogranicz częstotliwość prób ponownego wykrycia
            if current_time - last_reacquire_attempt > REACQUIRE_INTERVAL:
                print("Próba ponownego wykrycia osoby...")
                boxes_full, keypoints_batch_full = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=1280)
                last_reacquire_attempt = current_time
                
                if len(boxes_full) > 0:
                    # Znajdź najbliższą osobę do ostatniej znanej pozycji
                    prev_centroid = np.array([
                        (last_known_bbox[0] + last_known_bbox[2]) / 2,
                        (last_known_bbox[1] + last_known_bbox[3]) / 2
                    ])
                    
                    min_dist = float('inf')
                    best_idx = -1
                    
                    for i, box in enumerate(boxes_full):
                        centroid = np.array([
                            (box[0] + box[2]) / 2,
                            (box[1] + box[3]) / 2
                        ])
                        dist = euclidean(prev_centroid, centroid)
                        
                        if dist < min_dist and dist < TRACKING_THRESHOLD * 3:
                            min_dist = dist
                            best_idx = i
                    
                    if best_idx >= 0:
                        selected_bbox = boxes_full[best_idx]
                        target_keypoints = keypoints_batch_full[best_idx]
                        tracking_active = True
                        tracking_lost_count = 0
                        print(f"Ponownie wykryto osobę! (dystans: {min_dist:.1f} px)")
                        
                        # Zapis punktów kluczowych
                        if target_keypoints.size > 0 and target_keypoints.shape[0] >= 17:
                            all_keypoints.append(target_keypoints.flatten().tolist())
                        else:
                            all_keypoints.append([np.nan] * 34)
                        
                        # Kontynuuj do następnej klatki
                        frame_count += 1
                        continue
        
        # Jeśli śledzenie aktywne, spróbuj śledzić osobę
        if tracking_active:
            # Pełna detekcja okresowa
            if USE_FULL_FRAME_DETECTION and (current_time - last_reacquire_attempt > REACQUIRE_INTERVAL):
                boxes_full, keypoints_batch_full = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=1280)
                last_reacquire_attempt = current_time
                
                if len(boxes_full) > 0:
                    prev_centroid = np.array([
                        (selected_bbox[0] + selected_bbox[2]) / 2,
                        (selected_bbox[1] + selected_bbox[3]) / 2
                    ])
                    
                    min_dist = float('inf')
                    best_idx = -1
                    
                    for i, box in enumerate(boxes_full):
                        centroid = np.array([
                            (box[0] + box[2]) / 2,
                            (box[1] + box[3]) / 2
                        ])
                        dist = euclidean(prev_centroid, centroid)
                        
                        if dist < min_dist and dist < TRACKING_THRESHOLD * 2:
                            min_dist = dist
                            best_idx = i
                    
                    if best_idx >= 0:
                        selected_bbox = boxes_full[best_idx]
                        target_keypoints = keypoints_batch_full[best_idx]
                        
                        if target_keypoints.size > 0 and target_keypoints.shape[0] >= 17:
                            all_keypoints.append(target_keypoints.flatten().tolist())
                        else:
                            all_keypoints.append([np.nan] * 34)
                        
                        frame_count += 1
                        continue
            
            # Śledzenie w ROI
            roi_coords = get_roi(selected_bbox, frame.shape, ROI_EXPANSION)
            x_min, y_min, x_max, y_max = map(int, roi_coords)
            roi_frame = frame[y_min:y_max, x_min:x_max]
            
            if roi_frame.size == 0:
                all_keypoints.append([np.nan] * 34)
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    last_known_bbox = selected_bbox.copy()
                continue
                
            boxes, keypoints_batch = detect_people(roi_frame, CONFIDENCE_THRESHOLD)
            
            if len(boxes) == 0:
                all_keypoints.append([np.nan] * 34)
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    last_known_bbox = selected_bbox.copy()
                continue
            else:
                tracking_lost_count = 0
            
            # Przeskaluj wyniki
            for i in range(len(boxes)):
                boxes[i][0] += x_min
                boxes[i][1] += y_min
                boxes[i][2] += x_min
                boxes[i][3] += y_min
                
            for i in range(len(keypoints_batch)):
                for j in range(len(keypoints_batch[i])):
                    keypoints_batch[i][j][0] += x_min
                    keypoints_batch[i][j][1] += y_min
            
            # Jeśli wykryto tylko jedną osobę
            if len(boxes) == 1:
                target_keypoints = keypoints_batch[0]
                selected_bbox = boxes[0]
            else:
                prev_centroid = np.array([
                    (selected_bbox[0] + selected_bbox[2]) / 2,
                    (selected_bbox[1] + selected_bbox[3]) / 2
                ])
                
                min_dist = float('inf')
                best_idx = 0
                
                for i, box in enumerate(boxes):
                    centroid = np.array([
                        (box[0] + box[2]) / 2,
                        (box[1] + box[3]) / 2
                    ])
                    dist = euclidean(prev_centroid, centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                
                target_keypoints = keypoints_batch[best_idx]
                selected_bbox = boxes[best_idx]
            
            # Zapis punktów kluczowych
            if target_keypoints.size > 0 and target_keypoints.shape[0] >= 17:
                all_keypoints.append(target_keypoints.flatten().tolist())
            else:
                all_keypoints.append([np.nan] * 34)
        else:
            # Śledzenie nieaktywne - zapisz NaN
            all_keypoints.append([np.nan] * 34)
        
        # Podgląd śledzenia
        if SHOW_SKELETON and frame_count % 3 == 0:
            preview_frame = frame.copy()
            
            if tracking_active:
                # Narysuj ROI
                roi_coords = get_roi(selected_bbox, frame.shape, ROI_EXPANSION)
                x_min, y_min, x_max, y_max = map(int, roi_coords)
                cv2.rectangle(preview_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Narysuj szkielety
                if target_keypoints.size > 0 and target_keypoints.shape[0] >= 17:
                    draw_skeleton(preview_frame, target_keypoints, color=(0, 255, 0), thickness=3)
                    
                    # Zaznacz centroid
                    centroid = get_centroid(target_keypoints)
                    if not np.isnan(centroid).any():
                        cx, cy = int(centroid[0]), int(centroid[1])
                        cv2.circle(preview_frame, (cx, cy), 10, (0, 0, 255), -1)
            
            status_text = f"Status: {'AKTYWNE' if tracking_active else 'POSZUKIWANIE'}"
            status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
            
            cv2.putText(preview_frame, f"Klatka: {frame_count}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preview_frame, status_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(preview_frame, f"Utracone klatki: {tracking_lost_count}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(preview_window, preview_frame)
            cv2.waitKey(1)
        
        frame_count += 1

    cap.release()
    if SHOW_SKELETON:
        cv2.destroyWindow(preview_window)

    # Zapis do CSV
    df = pd.DataFrame(all_keypoints)
    output_path = output_folder / f"{video_file.stem}_clicked.csv"
    df.to_csv(output_path, index=False)
    print(f"Zapisano: {output_path}")
    
    # Oblicz statystyki
    total_frames_processed = len(all_keypoints)
    nan_frames = sum([1 for kp in all_keypoints if np.isnan(kp[0])])
    detection_rate = (1 - nan_frames / total_frames_processed) * 100
    
    print(f"Statystyki śledzenia:")
    print(f"- Przetworzone klatki: {total_frames_processed}")
    print(f"- Klatki z utraconym śledzeniem: {nan_frames}")
    print(f"- Współczynnik wykrywania: {detection_rate:.2f}%")
    print(f"Czas przetwarzania: {time.time()-start_time:.2f}s")

# Zamknięcie wszystkich okien na końcu
cv2.destroyAllWindows()