import pandas as pd
import re

# Ścieżka do pliku
csv_path = r'c:\Programs\studia\pracownia_problemowa\pracownia-problemowa\data\timePhoneData.csv'

# Wczytaj dane
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Nie znaleziono pliku: {csv_path}")
    exit()

def format_filename(filename):
    # Sprawdź czy plik zaczyna się od 'noaudio'
    if filename.startswith('noaudio'):
        # Wzorzec: noaudio + YYYY + MMDD + HHMMSS + NNN + .mp4
        # Przykład: noaudio20240217150242002.mp4
        match = re.match(r'noaudio(\d{4})(\d{4})(\d{6})(\d{3})\.mp4', filename)
        
        if match:
            year, date, time, number = match.groups()
            # Format docelowy: YYYY_MMDD_HHMMSS_NNN.MP4
            new_filename = f"{year}_{date}_{time}_{number}.MP4"
            return new_filename
    
    return filename

# Zastosuj funkcję do kolumny movieName
df['movieName'] = df['movieName'].apply(format_filename)

# Zapisz zmieniony plik
df.to_csv(csv_path, index=False)

print("Zaktualizowano nazwy plików w timePhoneData.csv")
print(df.head())
