import os
import csv


folder_path = "../../films"  

video_extensions = ('.mp4', '.mkv', '.avi', '.mov')

movies = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(video_extensions):
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace("_", "")
        new_path = os.path.join(folder_path, new_filename)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Zmieniono nazwę: {filename} -> {new_filename}")
        
        movies.append(new_filename)

csv_file = '../data/timePhoneData01.csv'

with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['movieName', 'startPhoneTime', 'serverTime'])
    
    for movie in movies:
        writer.writerow([movie]) 

print(f"\nZapisano dane do: {csv_file}")