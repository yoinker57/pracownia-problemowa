import pandas as pd
from datetime import datetime
from pathlib import Path
from moviepy.editor import VideoFileClip

# Wczytaj dane wejściowe
time_data = pd.read_csv("../data/timePhoneData.csv")
time_data['startPhoneTime'] = pd.to_datetime(time_data['startPhoneTime'])
time_data['serverTime'] = pd.to_datetime(time_data['serverTime'])

# Foldery
films_dir = Path("../../films")
output_dir = Path("../../trimmed_films")
output_dir.mkdir(exist_ok=True)

labeled_rows = []

device_folders = ["HONOR_8X", "huawei_mate"]

for _, row in time_data.iterrows():
    movie_name = row['movieName']
    start_phone_time = row['startPhoneTime']
    server_time = row['serverTime']
    data_file = str(row['dataFileName']).strip()
    date_str = server_time.strftime("%Y-%m-%d")

    data_files = [f.strip() for f in data_file.split('-')]
    for i, part in enumerate(data_files):
        df = None
        data_path = None

        for device_folder in device_folders:
            potential_path = Path(f"../../SkiTurnDetection/data/turns_with_styles/{device_folder}/{date_str}/{part}.csv")
            if potential_path.exists():
                data_path = potential_path
                try:
                    df = pd.read_csv(data_path)
                    df['time'] = pd.to_datetime(df['time'])
                    break
                except Exception as e:
                    print(f"Błąd wczytywania pliku {potential_path}: {e}")
                    continue

        if df is None:
            print(f"Brak pliku danych: {part}.csv w folderach {device_folders}")
            continue

        try:
            time_offset = start_phone_time - server_time
            filtered = df[df['Status'].isin(['START', 'STOP'])]

            if filtered.empty or 'START' not in filtered['Status'].values or 'STOP' not in filtered['Status'].values:
                print(f"{movie_name}: brak START lub STOP w części {part}")
                continue

            first_start = filtered[filtered['Status'] == 'START'].iloc[0]['time']
            last_stop = filtered[filtered['Status'] == 'STOP'].iloc[-1]['time']

            abs_first = first_start + time_offset
            abs_last = last_stop + time_offset

            trim_start_seconds = (abs_first - start_phone_time).total_seconds()
            trim_end_seconds = (abs_last - start_phone_time).total_seconds()

            part_name = f"{movie_name.replace('.mp4', '')}_{i}.mp4" if len(data_files) > 1 else movie_name

            style_row = filtered[filtered['Status'] == 'START'].iloc[0]
            labeled_rows.append({
                "movieName": 'trimmed_' + part_name,
                "STYLE": style_row.get("STYLE", ""),
                "SKIER_LEVEL": style_row.get("SKIER_LEVEL", ""),
                "SLOPE": style_row.get("SLOPE", ""),
                "start_sec": round(trim_start_seconds, 2),
                "end_sec": round(trim_end_seconds, 2)
            })

            print(f"{part_name} — przyciąć od {trim_start_seconds:.2f}s do {trim_end_seconds:.2f}s")

            # Przytnij wideo
            input_path = films_dir / movie_name
            output_path = output_dir / f"trimmed_{part_name}"

            if input_path.exists():
                clip = VideoFileClip(str(input_path)).subclip(trim_start_seconds, trim_end_seconds)
                clip.write_videofile(str(output_path), codec="libx264", audio=False)
                clip.close()
            else:
                print(f"Brak filmu: {input_path}")

        except Exception as e:
            print(f"Błąd podczas przetwarzania {data_path}: {e}")

# Zapisz etykiety
pd.DataFrame(labeled_rows).to_csv("labeledFilms.csv", index=False)
print("\nZapisano labeledFilms.csv")
