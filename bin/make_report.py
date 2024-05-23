import csv
import datetime
import glob
import os
from pyexcel import merge_all_to_a_book


def make_report(person_time, image_path, all_detect_count, isVideo):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    if isVideo:
        folder_path = os.path.join('reports/videos', formatted_time)
    else:
        folder_path = os.path.join('reports/photos', formatted_time)

    os.makedirs(folder_path, exist_ok=True)

    output_file = os.path.join(folder_path, f'person_report.csv')

    if isVideo:
        generate_report_video(person_time, output_file, image_path, all_detect_count)
    else:
        generate_report_photo(all_detect_count, output_file, image_path)


def generate_report_video(person_time, output_file, video_path, all_detect_count):
    with open(output_file, 'w', newline='', encoding='UTF8') as csvfile:
        if person_time is not None:
            fieldnames = ['Osoba', 'Całkowity czas (s)', 'Ilość wykonanych rzeczy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for person_name, (total_time, num_tasks) in person_time.items():
                writer.writerow({'Osoba': person_name, 'Całkowity czas (s)': total_time, 'Ilość wykonanych rzeczy': num_tasks})
        else:
            fieldnames = ['Czas', 'Największa ilość wykrytych osób']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for time_interval, max_detect_count in all_detect_count:
                writer.writerow({'Czas': time_interval, 'Największa ilość wykrytych osób': max_detect_count})

    merge_all_to_a_book(glob.glob(output_file), f"{output_file.rstrip('.csv')}.xlsx")
    os.startfile(output_file.rstrip(".csv")+".xlsx")

def generate_report_photo(all_detect_count, output_file, image_path):
    with open(output_file, 'w', newline='', encoding='UTF8') as csvfile:
        fieldnames = ['Nazwa Pliku', 'Łączna ilość wykrytych osób']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Nazwa Pliku': image_path, 'Łączna ilość wykrytych osób': all_detect_count})

    merge_all_to_a_book(glob.glob(output_file), f"{output_file.rstrip('.csv')}.xlsx")
    os.startfile(output_file.rstrip(".csv") + ".xlsx")