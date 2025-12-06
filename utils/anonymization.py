import cv2
import numpy as np
import os
import glob
import json

# Konfiguracja
DATA_DIR = r'c:\Programs\studia\pracownia_problemowa\data3'
STATE_FILE = 'anonymization_state.json'

# Zmienne globalne
frames = []
blur_masks = []
undo_stack = []
drawing = False
brush_radius = 12
blur_strength = 51
current_frame = 0
paused = True

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"video_index": 0, "frame_index": 0}

def save_state(video_index, frame_index):
    with open(STATE_FILE, 'w') as f:
        json.dump({"video_index": video_index, "frame_index": frame_index}, f)

def apply_blur(frame, mask):
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    return np.where(mask[..., None] == 255, blurred, frame)

def draw_progress_bar(frame, current, total, video_name):
    bar_height = 40
    bar_color = (0, 255, 0)
    bg_color = (50, 50, 50)
    text_color = (255, 255, 255)

    if total > 0:
        progress = int((current + 1) / total * frame.shape[1])
    else:
        progress = 0

    # Tło paska
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), bg_color, -1)
    # Pasek postępu
    cv2.rectangle(frame, (0, 0), (progress, bar_height), bar_color, -1)
    # Tekst
    text = f"{current + 1} / {total} | {video_name}"
    cv2.putText(frame, text, (10, bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

def mouse_callback(event, x, y, flags, param):
    global drawing, brush_radius, undo_stack, current_frame, blur_masks

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Zapisz stan przed rysowaniem (do undo)
        if current_frame < len(blur_masks):
            undo_stack.append(blur_masks[current_frame].copy())
            if len(undo_stack) > 20:  # limit historii
                undo_stack.pop(0)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if current_frame < len(blur_masks):
            cv2.circle(blur_masks[current_frame], (x, y), brush_radius, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_radius = min(100, brush_radius + 2)
        else:
            brush_radius = max(2, brush_radius - 2)
        print(f"Rozmiar pędzla: {brush_radius}")

def process_video(video_path, start_frame_idx):
    global frames, blur_masks, undo_stack, current_frame, paused
    
    print(f"Przetwarzanie: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    blur_masks = []
    undo_stack = []
    current_frame = start_frame_idx
    paused = True
    
    print("Ładowanie klatek...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        blur_masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
    cap.release()
    
    if not frames:
        print("Błąd: Nie udało się wczytać klatek (plik może być uszkodzony lub pusty).")
        return 'next', 0

    print(f"Załadowano {len(frames)} klatek.")
    
    # Zabezpieczenie, gdyby zapisany stan wykraczał poza długość filmu
    if current_frame >= len(frames):
        current_frame = 0

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)
    
    action = 'quit' 
    
    while True:
        frame_disp = frames[current_frame].copy()
        mask = blur_masks[current_frame]
        blurred_frame = apply_blur(frame_disp, mask)
        draw_progress_bar(blurred_frame, current_frame, len(frames), os.path.basename(video_path))

        cv2.imshow("Video", blurred_frame)
        key = cv2.waitKey(30)

        if key == ord('q'):
            action = 'quit'
            break
        elif key == ord(' '):  # spacja
            paused = not paused
        elif key == ord('v'):  # ← klatka wstecz
            current_frame = max(0, current_frame - 1)
            undo_stack.clear()
        elif key == ord('b'):  # → klatka dalej
            current_frame = min(len(frames) - 1, current_frame + 1)
            undo_stack.clear()
        elif key == ord('z'):  # cofnij blur
            if undo_stack:
                blur_masks[current_frame] = undo_stack.pop()
                print("Cofnięto ostatni blur")
            else:
                print("Brak zmian do cofnięcia")
        elif key == ord('s'):  # zapisz i następny
            print("Zapisuję film...")
            height, width, _ = frames[0].shape
            
            # Tworzenie nazwy pliku wyjściowego (np. video_anonymized.mp4)
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_anonymized{ext}"
            
            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  30, (width, height))

            for i in range(len(frames)):
                f = frames[i].copy()
                m = blur_masks[i]
                out.write(apply_blur(f, m))

            out.release()
            print(f"Zapisano jako {output_path}")
            action = 'next'
            break
        
        if not paused:
            current_frame += 1
            if current_frame >= len(frames):
                current_frame = len(frames) - 1
                paused = True # Zatrzymaj na końcu
            undo_stack.clear()
            
    cv2.destroyAllWindows()
    return action, current_frame

def main():
    state = load_state()
    video_idx = state['video_index']
    frame_idx = state['frame_index']
    
    print(f"Szukam plików w: {DATA_DIR}")
    # Rekurencyjne szukanie wszystkich .mp4
    all_files = glob.glob(os.path.join(DATA_DIR, '**', '*.mp4'), recursive=True)
    
    # Filtrowanie plików, żeby nie przetwarzać już zanonimizowanych (jeśli są w tym samym folderze)
    video_files = sorted([f for f in all_files if '_anonymized' not in f])
    
    if not video_files:
        print("Nie znaleziono plików wideo.")
        return

    print(f"Znaleziono {len(video_files)} plików do przetworzenia.")
    
    while video_idx < len(video_files):
        video_path = video_files[video_idx]
        
        # Jeśli wczytujemy film, na którym skończyliśmy, przywracamy klatkę.
        # W przeciwnym razie startujemy od 0.
        current_start_frame = frame_idx if video_idx == state['video_index'] else 0
        
        # Resetujemy frame_idx w stanie, żeby kolejne filmy startowały od 0, 
        # chyba że znowu przerwiemy (wtedy zaktualizujemy stan przy wyjściu).
        
        action, last_frame = process_video(video_path, current_start_frame)
        
        if action == 'next':
            video_idx += 1
            frame_idx = 0 # Następny film od początku
            save_state(video_idx, 0)
        elif action == 'quit':
            save_state(video_idx, last_frame)
            print("Zapisano stan i zakończono.")
            break
            
    if video_idx >= len(video_files):
        print("Wszystkie filmy zostały przetworzone!")
        # Opcjonalnie: reset stanu
        # save_state(0, 0) 

if __name__ == "__main__":
    main()
