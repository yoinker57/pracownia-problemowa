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
REACQUIRE_INTERVAL = 1.0
REACQUIRE_MAX_LOST_TIME = 2
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
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))
            cv2.line(frame, start, end, color, thickness)

    for point in keypoints:
        if not np.isnan(point).any():
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

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

# --- Argumenty ---
parser = argparse.ArgumentParser(description="Process ski videos.")
parser.add_argument(
    "--video_folder",
    type=str,
    default="../../trimmed_films",
    help="Path to the folder containing ski videos"
)
args = parser.parse_args()
video_folder = Path(args.video_folder)
video_files = list(video_folder.glob("*.mp4"))

output_folder = Path("../pose_outputs_loss")
output_folder.mkdir(exist_ok=True)

# --- Zmienne globalne ---
click_position = []
selected_bbox = None
confirmed = False
current_frame_idx = 0
pause = False

waiting_for_reselect = False
lost_tracking_start_time = None

def on_mouse(event, x, y, flags, param):
    global click_position
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Kliknięcie na {x}, {y}")
        click_position.append((x, y))

def update_frame_pos(pos):
    global current_frame_idx
    current_frame_idx = pos

# --- Przetwarzanie plików ---
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
    current_frame_idx = 0

    # --- Faza wyboru początkowej osoby ---
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
            pause = False
        elif key == ord('.'):
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
            pause = True
        elif key == ord(','):
            current_frame_idx = max(current_frame_idx - 1, 0)
            pause = True

        if click_position and len(boxes) > 0:
            click_x, click_y = click_position.pop(0)

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
    # Zapisujemy klatki, w których nie wykryto osoby jako NaN
    empty_kp = [np.nan] * 34
    all_keypoints = [empty_kp] * int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Uzupełniamy klatki już przetworzone
    
    frame_count = 0
    tracking_lost_count = 0
    tracking_active = True
    last_known_bbox = selected_bbox.copy()
    last_reacquire_attempt = time.time()
    
    # Zmienne dla trybu reselect
    reselect_bbox = None
    reselect_idx = -1
    outer_break = False  # DODANA DEKLARACJA ZMIENNEJ

    if SHOW_SKELETON:
        preview_window = "Podglad sledzenia"
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 800, 600)
        cv2.setMouseCallback(preview_window, on_mouse)

    start_time = time.time()

    # Przewiń do początku
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame = 0

    while current_frame < total_frames and not outer_break:  # DODANY WARUNEK outer_break
        ret, frame = cap.read()
        if not ret:
            # Jeśli nie udało się wczytać klatki, dodajemy pusty wpis i kontynuujemy
            all_keypoints.append(empty_kp)
            current_frame += 1
            continue
            
        # Aktualizacja numeru klatki
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # Jeśli to nowa klatka, upewnij się że mamy miejsce w liście
        while len(all_keypoints) <= current_frame:
            all_keypoints.append(empty_kp)

        current_time = time.time()
        
        # --- Tryb ponownego wyboru (pełna funkcjonalność) ---
        if waiting_for_reselect:
            # Wykryj osoby na bieżącej klatce
            boxes_full, keypoints_batch_full = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=1280)
            
            # Wewnętrzna pętla - pokazujemy tę samą klatkę aż do wyboru osoby
            while True:
                display_frame = frame.copy()
                
                # Rysujemy wszystkie osoby
                for i, box in enumerate(boxes_full):
                    x_min, y_min, x_max, y_max = map(int, box[:4])
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"#{i}", (x_min+5, y_min+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Obsługa kliknięcia - zapamiętujemy wybór
                if click_position:
                    click_x, click_y = click_position.pop(0)
                    
                    min_dist = float('inf')
                    closest_idx = -1
                    for i, box in enumerate(boxes_full):
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = i

                    if closest_idx >= 0:
                        reselect_bbox = boxes_full[closest_idx]
                        reselect_idx = closest_idx
                        print(f"Wybrano osobę #{closest_idx} do ponownego śledzenia")
                
                # Rysujemy wybraną osobę (jeśli została wybrana)
                if reselect_bbox is not None:
                    x_min, y_min, x_max, y_max = map(int, reselect_bbox)
                    cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

                # Instrukcje
                info_y = 30
                cv2.putText(display_frame, "TRACKING UTRACONY - Wybierz osobe ponownie", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, "Kliknij osobe i nacisnij [C] by zatwierdzic", 
                           (10, info_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, "[Spacja]: Play/Pause | [Q]: Wyjscie", 
                           (10, info_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.imshow(preview_window, display_frame)
                
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    outer_break = True  # USTAW FLAGĘ PRZERWANIA
                    break
                elif key == ord(' '):
                    # Pauza w trybie reselect
                    pass
                elif key == ord('c') and reselect_bbox is not None:
                    selected_bbox = reselect_bbox
                    tracking_active = True
                    waiting_for_reselect = False
                    lost_tracking_start_time = None
                    print("Wznowiono śledzenie po reselekcji")
                    
                    # Znajdź keypoints dla wybranej osoby
                    if reselect_idx < len(keypoints_batch_full) and keypoints_batch_full[reselect_idx].size > 0 and keypoints_batch_full[reselect_idx].shape[0] >= 17:
                        kp = keypoints_batch_full[reselect_idx].flatten().tolist()
                    else:
                        kp = empty_kp  # Używamy pustego keypointa
                    
                    # Aktualizuj wpis dla bieżącej klatki
                    if current_frame < len(all_keypoints):
                        all_keypoints[current_frame] = kp
                    else:
                        all_keypoints.append(kp)
                    
                    # Wyjdź z wewnętrznej pętli
                    break
                elif key == ord('c') and reselect_bbox is None:
                    print("Najpierw wybierz osobę klikając na nią!")
            
            # Jeśli użytkownik wybrał 'q', przerwij główną pętlę
            if outer_break:
                break
            
            # Przejdź do następnej klatki
            current_frame += 1
            continue

        # --- Automatyczne ponowne wykrywanie ---
        if not tracking_active:
            if lost_tracking_start_time is None:
                lost_tracking_start_time = current_time
            elapsed = current_time - lost_tracking_start_time

            if elapsed > REACQUIRE_MAX_LOST_TIME:
                print("Tracking utracony > 4s, przechodzę w tryb ponownego wyboru...")
                waiting_for_reselect = True
                # Resetujemy wybór dla reselect
                reselect_bbox = None
                reselect_idx = -1
                # Przejdź do następnej klatki
                current_frame += 1
                continue

            if current_time - last_reacquire_attempt > REACQUIRE_INTERVAL:
                print("Próba ponownego wykrycia osoby...")
                boxes_full, keypoints_batch_full = detect_people(frame, CONFIDENCE_THRESHOLD, imgsz=1280)
                last_reacquire_attempt = current_time

                if len(boxes_full) > 0:
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
                        lost_tracking_start_time = None
                        print(f"Ponownie wykryto osobę! (dystans: {min_dist:.1f} px)")

                        if target_keypoints.size > 0 and target_keypoints.shape[0] >= 17:
                            kp = target_keypoints.flatten().tolist()
                        else:
                            kp = empty_kp
                        
                        # Aktualizuj wpis dla bieżącej klatki
                        if current_frame < len(all_keypoints):
                            all_keypoints[current_frame] = kp
                        else:
                            all_keypoints.append(kp)
                else:
                    # Nie wykryto osób - aktualizuj bieżącą klatkę jako pustą
                    if current_frame < len(all_keypoints):
                        all_keypoints[current_frame] = empty_kp
                    else:
                        all_keypoints.append(empty_kp)
                
                # Przejdź do następnej klatki
                current_frame += 1
                continue

        # --- Normalne śledzenie ---
        if tracking_active:
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
                            kp = target_keypoints.flatten().tolist()
                        else:
                            kp = empty_kp
                        
                        # Aktualizuj wpis dla bieżącej klatki
                        if current_frame < len(all_keypoints):
                            all_keypoints[current_frame] = kp
                        else:
                            all_keypoints.append(kp)
                        
                        # Przejdź do następnej klatki
                        current_frame += 1
                        continue

            roi_coords = get_roi(selected_bbox, frame.shape, ROI_EXPANSION)
            x_min, y_min, x_max, y_max = map(int, roi_coords)
            roi_frame = frame[y_min:y_max, x_min:x_max]

            if roi_frame.size == 0:
                # Aktualizuj wpis dla bieżącej klatki
                if current_frame < len(all_keypoints):
                    all_keypoints[current_frame] = empty_kp
                else:
                    all_keypoints.append(empty_kp)
                
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    lost_tracking_start_time = current_time
                
                # Przejdź do następnej klatki
                current_frame += 1
                continue

            roi_results = model(roi_frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)
            if len(roi_results) == 0 or len(roi_results[0].boxes) == 0:
                # Aktualizuj wpis dla bieżącej klatki
                if current_frame < len(all_keypoints):
                    all_keypoints[current_frame] = empty_kp
                else:
                    all_keypoints.append(empty_kp)
                
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    lost_tracking_start_time = current_time
                
                # Przejdź do następnej klatki
                current_frame += 1
                continue

            roi_box = roi_results[0].boxes.xyxy.cpu().numpy()
            roi_keypoints = roi_results[0].keypoints.xy.cpu().numpy()

            if len(roi_box) == 0:
                # Aktualizuj wpis dla bieżącej klatki
                if current_frame < len(all_keypoints):
                    all_keypoints[current_frame] = empty_kp
                else:
                    all_keypoints.append(empty_kp)
                
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    lost_tracking_start_time = current_time
                
                # Przejdź do następnej klatki
                current_frame += 1
                continue

            # Wybierz najlepszy box/keypoints na podstawie odległości centroidów
            prev_centroid = np.array([
                (selected_bbox[0] + selected_bbox[2]) / 2,
                (selected_bbox[1] + selected_bbox[3]) / 2
            ])

            min_dist = float('inf')
            best_idx = -1
            for i, box in enumerate(roi_box):
                cx = (box[0] + box[2]) / 2 + x_min
                cy = (box[1] + box[3]) / 2 + y_min
                dist = euclidean(prev_centroid, np.array([cx, cy]))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i

            if best_idx == -1:
                # Aktualizuj wpis dla bieżącej klatki
                if current_frame < len(all_keypoints):
                    all_keypoints[current_frame] = empty_kp
                else:
                    all_keypoints.append(empty_kp)
                
                tracking_lost_count += 1
                if tracking_lost_count > 10:
                    print("Utracono śledzenie - przechodzę w tryb ponownego wykrywania")
                    tracking_active = False
                    lost_tracking_start_time = current_time
                
                # Przejdź do następnej klatki
                current_frame += 1
                continue

            # Przesuń keypoints do koordynat globalnych ramki
            kp = roi_keypoints[best_idx]
            kp[:, 0] += x_min
            kp[:, 1] += y_min

            # Aktualizuj wpis dla bieżącej klatki
            kp_list = kp.flatten().tolist()
            if current_frame < len(all_keypoints):
                all_keypoints[current_frame] = kp_list
            else:
                all_keypoints.append(kp_list)

            # Aktualizuj selected_bbox
            selected_bbox = roi_box[best_idx] + np.array([x_min, y_min, x_min, y_min])
            last_known_bbox = selected_bbox.copy()
            tracking_lost_count = 0

            if SHOW_SKELETON:
                preview_frame = frame.copy()
                draw_skeleton(preview_frame, kp, color=(0, 255, 0), thickness=2)
                cv2.rectangle(preview_frame, (int(selected_bbox[0]), int(selected_bbox[1])), 
                              (int(selected_bbox[2]), int(selected_bbox[3])), (0, 0, 255), 2)
                cv2.imshow(preview_window, preview_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            outer_break = True  # USTAW FLAGĘ PRZERWANIA
        
        # Przejdź do następnej klatki
        current_frame += 1

    # Upewnij się, że mamy wpisy dla wszystkich klatek
    while len(all_keypoints) < total_frames:
        all_keypoints.append(empty_kp)

    cap.release()
    if SHOW_SKELETON:
        cv2.destroyWindow(preview_window)

    # Zapisz dane do CSV (bez "_pose" w nazwie)
    column_names = [str(i) for i in range(1, 35)]
    df = pd.DataFrame(all_keypoints, columns=column_names)
    csv_path = output_folder / (video_file.stem + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Zapisano wyniki do {csv_path}")
    
    # Oblicz procent wykrycia
    detected_frames = sum(1 for kp in all_keypoints if not all(pd.isna(x) for x in kp))
    detection_rate = 100 * (detected_frames / total_frames) if total_frames > 0 else 0
    
    # Ścieżka do pliku statystyk
    stats_path = output_folder / "stats.csv"
    
    # Utwórz DataFrame z wpisem
    stats_entry = pd.DataFrame([{
        "file": video_file.stem,
        "percent_of_detection": round(detection_rate, 2)
    }])
    
    # Dopisz lub utwórz plik
    if stats_path.exists():
        stats_entry.to_csv(stats_path, mode='a', header=False, index=False)
    else:
        stats_entry.to_csv(stats_path, mode='w', header=True, index=False)
    
    print(f"Zaktualizowano statystyki w pliku: {stats_path}")

print("Przetwarzanie zakończone.")