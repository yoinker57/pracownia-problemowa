import cv2
import numpy as np

video_path = 'movie.mp4'  # <-- zmień jeśli trzeba
cap = cv2.VideoCapture(video_path)

frames = []
blur_masks = []
undo_stack = []

paused = True
current_frame = 0
drawing = False
brush_radius = 12
blur_strength = 51  # większa wartość = mocniejszy blur (musi być nieparzysta)

# Załaduj wszystkie klatki
print("Ładowanie klatek...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    blur_masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))  # maska dla każdej klatki
print(f"Załadowano {len(frames)} klatek.")
cap.release()

def apply_blur(frame, mask):
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    return np.where(mask[..., None] == 255, blurred, frame)

def draw_progress_bar(frame, current, total):
    bar_height = 20
    bar_color = (0, 255, 0)
    bg_color = (50, 50, 50)
    text_color = (255, 255, 255)

    progress = int((current + 1) / total * frame.shape[1])

    # Tło paska
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), bg_color, -1)
    # Pasek postępu
    cv2.rectangle(frame, (0, 0), (progress, bar_height), bar_color, -1)
    # Tekst (np. "123 / 2703")
    text = f"{current + 1} / {total}"
    cv2.putText(frame, text, (10, bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

# Myszka
def mouse_callback(event, x, y, flags, param):
    global drawing, brush_radius, undo_stack

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Zapisz stan przed rysowaniem (do undo)
        undo_stack.append(blur_masks[current_frame].copy())
        if len(undo_stack) > 20:  # limit historii
            undo_stack.pop(0)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(blur_masks[current_frame], (x, y), brush_radius, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            brush_radius = min(100, brush_radius + 2)
        else:
            brush_radius = max(2, brush_radius - 2)
        print(f"Rozmiar pędzla: {brush_radius}")

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while True:
    frame = frames[current_frame].copy()
    mask = blur_masks[current_frame]
    blurred_frame = apply_blur(frame, mask)
    draw_progress_bar(blurred_frame, current_frame, len(frames))

    cv2.imshow("Video", blurred_frame)
    key = cv2.waitKey(30)

    if key == ord('q'):
        break
    elif key == ord(' '):  # spacja
        paused = not paused
    elif key == ord('v'):  # ← klatka wstecz
        current_frame = max(0, current_frame - 1)
        undo_stack.clear()
    elif key == ord('b'):  # → klatka dalej
        current_frame = min(len(frames) - 1, current_frame + 1)
        print(f'current frame: {current_frame}')
        undo_stack.clear()
    elif key == ord('z'):  # cofnij blur
        if undo_stack:
            blur_masks[current_frame] = undo_stack.pop()
            print("Cofnięto ostatni blur")
        else:
            print("Brak zmian do cofnięcia")
    elif key == ord('s'):  # zapisz
        print("Zapisuję film...")
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter('movie_a2.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30, (width, height))

        for i in range(len(frames)):
            frame = frames[i].copy()
            mask = blur_masks[i]
            out.write(apply_blur(frame, mask))

        out.release()
        print("Zapisano jako output_blurred.mp4")
    elif not paused:
        current_frame += 1
        if current_frame >= len(frames):
            current_frame = len(frames) - 1
        undo_stack.clear()

cv2.destroyAllWindows()
