import os
import cv2
import numpy as np
from pathlib import Path

def get_video_properties(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    file_size = os.path.getsize(video_path)
    bitrate = (file_size * 8) / duration if duration > 0 else 0
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "bitrate": bitrate,
        "file_size": file_size
    }

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compare_videos(dir1, dir2):
    print(f"Porównywanie filmów z '{dir2}' (badany) do '{dir1}' (referencyjny)...\n")
    
    issues_found = []
    
    for root, dirs, files in os.walk(dir2):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                path2 = Path(root) / file
                rel_path = path2.relative_to(dir2)
                path1 = Path(dir1) / rel_path
                
                if not path1.exists():
                    # print(f"Brak odpowiednika w folderze referencyjnym: {rel_path}")
                    continue
                
                props1 = get_video_properties(path1)
                props2 = get_video_properties(path2)
                
                if not props1 or not props2:
                    print(f"Błąd odczytu pliku: {rel_path}")
                    continue
                
                # Porównanie metadanych
                reasons = []
                
                # 1. Rozdzielczość
                pixels1 = props1['width'] * props1['height']
                pixels2 = props2['width'] * props2['height']
                if pixels2 < pixels1 * 0.9: # 10% tolerancji
                    reasons.append(f"Spadek rozdzielczości: {props1['width']}x{props1['height']} -> {props2['width']}x{props2['height']}")
                
                # 2. Bitrate
                if props2['bitrate'] < props1['bitrate'] * 0.5: # 50% spadku bitrate
                    reasons.append(f"Znaczny spadek bitrate: {int(props1['bitrate']/1024)} kbps -> {int(props2['bitrate']/1024)} kbps")
                
                # 3. Czas trwania (czy to ten sam film?)
                if abs(props1['duration'] - props2['duration']) > 1.0:
                    reasons.append(f"Różnica w czasie trwania: {props1['duration']:.2f}s vs {props2['duration']:.2f}s (pomijam analizę obrazu)")
                else:
                    # 4. Analiza obrazu (PSNR na środkowej klatce)
                    try:
                        cap1 = cv2.VideoCapture(str(path1))
                        cap2 = cv2.VideoCapture(str(path2))
                        
                        # Ustaw na środek
                        mid_frame = int(props1['duration'] * props1['fps'] / 2)
                        cap1.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                        
                        ret1, frame1 = cap1.read()
                        ret2, frame2 = cap2.read()
                        
                        if ret1 and ret2:
                            # Jeśli rozmiary są różne, przeskaluj frame2 do frame1
                            if frame1.shape != frame2.shape:
                                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
                            
                            psnr = calculate_psnr(frame1, frame2)
                            if psnr < 30: # Poniżej 30dB to zazwyczaj widoczna strata jakości
                                reasons.append(f"Niska jakość wizualna (PSNR: {psnr:.2f} dB)")
                        
                        cap1.release()
                        cap2.release()
                    except Exception as e:
                        print(f"Błąd podczas analizy obrazu {rel_path}: {e}")

                if reasons:
                    print(f"⚠️  PROBLEM Z PLIKIEM: {rel_path}")
                    for r in reasons:
                        print(f"   - {r}")
                    print("-" * 40)
                    issues_found.append(rel_path)

    if not issues_found:
        print("\nNie znaleziono znaczących problemów z jakością.")
    else:
        print(f"\nZnaleziono {len(issues_found)} plików o niższej jakości.")

if __name__ == "__main__":
    # Ścieżki
    data_dir = Path(r"c:\Programs\studia\pracownia_problemowa\data")
    data2_dir = Path(r"c:\Programs\studia\pracownia_problemowa\data2")
    
    if data_dir.exists() and data2_dir.exists():
        compare_videos(data_dir, data2_dir)
    else:
        print("Nie znaleziono folderów data lub data2.")
