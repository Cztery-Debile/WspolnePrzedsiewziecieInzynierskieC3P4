import numpy as np
from imgbeddings import imgbeddings
from PIL import Image as Image1
import psycopg2
import os
import ast


def get_names_list():
    conn = psycopg2.connect(
        "postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require")

    # Ścieżka do katalogu zawierającego zdjęcia do porównania
    compare_folder = "../face_detection/compare"

    # Inicjalizacja pustej listy na imiona
    names_list = []

    # Przetwórz każde zdjęcie w katalogu porównawczym
    for filename in os.listdir(compare_folder):
        if filename.startswith("face_") and filename.endswith(".png"):
            # Pobierz ID twarzy z nazwy pliku
            face_id = int(filename.split("_")[1].split(".")[0])

            file_path = os.path.join(compare_folder, filename)
            img = Image1.open(file_path)
            ibed = imgbeddings()
            embedding = ibed.to_embeddings(img)

            cur = conn.cursor()
            string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
            cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
            rows = cur.fetchall()
            for row in rows:
                embedded_image = np.array(ast.literal_eval(row[1]))
                norm_distance = np.linalg.norm(embedded_image - embedding[0])
                name = row[2]
                print(norm_distance)
                # Dodaj nowe imię i ID do listy
                if norm_distance < 10.2:
                    # Sprawdź, czy na liście nie ma już tej samej nazwy
                    if all(name != existing_name for existing_name, _ in names_list):
                        # Jeśli nazwa nie istnieje na liście, dodaj nową parę (name, face_id)
                        new_entry = (name, face_id)
                        names_list.append(new_entry)
                    else:
                        # Jeśli nazwa już istnieje na liście, wykonaj odpowiednie działania (np. zignoruj)
                        pass
            cur.close()

    conn.close()
    print(names_list)
    return  names_list

get_names_list()
