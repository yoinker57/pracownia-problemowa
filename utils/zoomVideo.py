import cv2
import numpy as np
import os
from collections import defaultdict

# Konfiguracja
INPUT_VIDEO = "trimmed_noaudio20240317130227004_1.mp4"
ZOOM_FACTOR = 2.0
BOX_COLOR = (0, 255, 0)

def create_tracker():
    if hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    raise AttributeError("Brak trackera CSRT – zainstaluj opencv-contrib-python.")

class PersonTracker:
    def __init__(self):
        self.trackers: list[tuple[int, cv2.Tracker]] = []
        self.next_id = 0

    def add_person(self, frame, bbox):
        trk = create_tracker()
        trk.init(frame, bbox)
        self.trackers.append((self.next_id, trk))
        self.next_id += 1

    def update(self, frame):
        active = {}
        dead = []
        for pid, trk in self.trackers:
            ok, box = trk.update(frame)
            if ok:
                active[pid] = box
            else:
                dead.append((pid, trk))
        for d in dead:
            self.trackers.remove(d)
        return active

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Nie można otworzyć pliku {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs("zoomed_videos", exist_ok=True)

    person_writers = {}
    zoom_sizes = {}
    tracker = PersonTracker()

    selected_bbox = None
    selecting, paused = False, True

    def select(event, x, y, flags, _):
        nonlocal selected_bbox, selecting
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            selected_bbox = [x, y, 0, 0]
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            selected_bbox[2] = x - selected_bbox[0]
            selected_bbox[3] = y - selected_bbox[1]
        elif event == cv2.EVENT_LBUTTONUP and selecting:
            selecting = False
            selected_bbox[2] = x - selected_bbox[0]
            selected_bbox[3] = y - selected_bbox[1]

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", select)

    ret, current = cap.read()
    if not ret:
        raise SystemExit("Brak klatki początkowej.")
    display = current.copy()

    print("Instrukcja:")
    print("  • Zaznacz osobę myszą (przytrzymaj LPM)")
    print("  • Spacja  – dodaj osobę do śledzenia")
    print("  • S       – start / pauza")
    print("  • Q       – zakończ")

    while True:
        if selecting and selected_bbox:
            x, y, w, h = selected_bbox
            x1, x2 = (x, x + w) if w >= 0 else (x + w, x)
            y1, y2 = (y, y + h) if h >= 0 else (y + h, y)
            cv2.rectangle(display, (x1, y1), (x2, y2), BOX_COLOR, 2)

        active_boxes = tracker.update(current)
        for pid, box in active_boxes.items():
            x, y, w, h = map(int, box)
            cv2.rectangle(display, (x, y), (x + w, y + h), BOX_COLOR, 2)
            cv2.putText(display, f"ID:{pid}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)

        cv2.imshow("Tracking", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(" "):
            if selected_bbox and not selecting:
                x, y, w, h = selected_bbox
                x1 = min(x, x + w); y1 = min(y, y + h)
                w = abs(w); h = abs(h)
                if w > 10 and h > 10:
                    tracker.add_person(current, (x1, y1, w, h))
                    print(f"➕ Dodano osobę ID:{tracker.next_id - 1}")
                selected_bbox = None
                display = current.copy()
        elif key == ord("s"):
            paused = not paused
            print("▶️ Start" if not paused else "⏸️ Pauza")
        elif key == ord("q"):
            break

        if not paused:
            ret, current = cap.read()
            if not ret:
                break
            display = current.copy()
            frame_h, frame_w = current.shape[:2]

            for pid, box in active_boxes.items():
                x, y, w, h = map(int, box)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_w, x + w)
                y2 = min(frame_h, y + h)
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue

                roi = current[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                if pid not in zoom_sizes:
                    zoom_w = int((x2 - x1) * ZOOM_FACTOR)
                    zoom_h = int((y2 - y1) * ZOOM_FACTOR)
                    zoom_sizes[pid] = (zoom_w, zoom_h)
                else:
                    zoom_w, zoom_h = zoom_sizes[pid]

                zoomed = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

                if pid not in person_writers:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = f"zoomed_videos/person_{pid}.mp4"
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (zoom_w, zoom_h))
                    if not writer.isOpened():
                        print(f"❌ Nie można utworzyć zapisu dla {out_path}")
                        continue
                    person_writers[pid] = writer
                    print(f"▶️  Utworzono plik {out_path}")

                person_writers[pid].write(zoomed)
                # Debug info: print(f"✔️ Zapisano klatkę dla ID:{pid}")

    cap.release()
    for wr in person_writers.values():
        wr.release()
    cv2.destroyAllWindows()
    print("✔️  Gotowe – pliki zapisane w zoomed_videos/")

if __name__ == "__main__":
    main()
