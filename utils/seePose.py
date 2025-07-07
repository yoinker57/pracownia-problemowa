import cv2
import pandas as pd
from pathlib import Path
import numpy as np

# Folder z video
video_folder = Path("../../trimmed_films")

# Foldery z CSV (możesz dopisać więcej)
csv_folders = [
    Path("../pose_outputs_auto"),
    Path("../pose_outputs_clicked"),
    Path("../pose_outputs_loss"),
    # dodaj kolejne foldery z CSV
]

# Kolory do rysowania (BGR) - musi być tyle samo lub więcej niż folderów csv
colors = [
    (0, 255, 0),    # zielony
    (0, 0, 255),    # czerwony
    (255, 0, 0),    # niebieski
    (0, 255, 255),  # żółty
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, points, color):
    for x, y in points:
        if x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for start, end in SKELETON_CONNECTIONS:
        if points[start][0] > 0 and points[start][1] > 0 and \
           points[end][0] > 0 and points[end][1] > 0:
            cv2.line(frame,
                     (int(points[start][0]), int(points[start][1])),
                     (int(points[end][0]), int(points[end][1])),
                     color, 2)

# Zbierz listę video do przetworzenia
video_files = list(video_folder.glob("*.mp4"))

for video_path in video_files:
    video_name = video_path.stem
    print(f"Przetwarzanie wideo: {video_name}")

    # Wczytaj CSV z każdego folderu, jeśli istnieje plik o tej samej nazwie
    csv_dfs = []
    for i, folder in enumerate(csv_folders):
        csv_path = folder / f"{video_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None)
            csv_dfs.append((df, colors[i]))
            print(f"  Załadowano CSV: {csv_path}")
        else:
            print(f"  Brak CSV: {csv_path}")

    if not csv_dfs:
        print("  Brak plików CSV do porównania, pomijam wideo.")
        continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Nie można otworzyć wideo: {video_path}")
        continue

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Nakładamy szkielety z każdego CSV
        for df, color in csv_dfs:
            if frame_idx >= len(df):
                continue
            row = df.iloc[frame_idx]
            points = [(row[i], row[i+1]) for i in range(0, len(row), 2)]
            draw_skeleton(frame, points, color)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Opcjonalnie legendę
        y0 = 60
        for idx, (_, color) in enumerate(csv_dfs):
            cv2.putText(frame, f"Method {idx+1}", (10, y0 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Compare Skeletons", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1

    cap.release()

cv2.destroyAllWindows()
