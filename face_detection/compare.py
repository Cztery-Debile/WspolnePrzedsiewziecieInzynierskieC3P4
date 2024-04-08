import numpy as np
from imgbeddings import imgbeddings
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
from PIL import Image as Image1
import psycopg2
import psycopg2.pool
import os
import ast
import time
from concurrent.futures import ThreadPoolExecutor

MIN_CONN = 1
MAX_CONN = 5


def get_connection_pool():
    return psycopg2.pool.ThreadedConnectionPool(
        MIN_CONN,
        MAX_CONN,
        "postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require"
    )


def process_image(file_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB mode

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Preprocess the image
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    t = time.time()

    # Pass the preprocessed image through the model to obtain the embedding
    with torch.no_grad():
        embedding = resnet(img_tensor).cpu().numpy()  # Move the embedding to CPU and convert to numpy array

    end_time = time.time()
    execution_time = end_time - t
    print("Czas wykonania matematyka:", execution_time, "sekund")

    return embedding, file_path

def process_results(results):
    names_list = []
    conn_pool = get_connection_pool()
    try:
        with conn_pool.getconn() as conn:
            for embedding, file_path in results:
                with conn.cursor() as cur:
                    string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
                    cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
                    rows = cur.fetchall()
                    for row in rows:

                        embedded_image = np.array(ast.literal_eval(row[1]))
                        norm_distance = np.linalg.norm(embedded_image - embedding[0])
                        name = row[2]
                    #    print(norm_distance)
                        # Dodaj nowe imię i ID do listy
                        if norm_distance < 10.2:
                            # Sprawdź, czy na liście nie ma już tej samej nazwy

                            if all(name != existing_name for existing_name, _ in names_list):
                                # Jeśli nazwa nie istnieje na liście, dodaj nową parę (name, face_id)
                                face_id = int(os.path.basename(file_path).split("_")[1].split(".")[0])
                                new_entry = (name, face_id)
                                names_list.append(new_entry)
                            else:
                                # Jeśli nazwa już istnieje na liście, wykonaj odpowiednie działania (np. zignoruj)
                                pass

    finally:
        conn_pool.closeall()
    return names_list


def get_names_list():
    start_time = time.time()
    # Ścieżka do katalogu zawierającego zdjęcia do porównania
    compare_folder = "../face_detection/compare"

    try:
        # Utwórz pulę wątków
        with ThreadPoolExecutor() as executor:
            # Zgłoś zadania dla każdego pliku obrazu w katalogu
            futures = [executor.submit(process_image, os.path.join(compare_folder, filename))
                       for filename in os.listdir(compare_folder)
                       if filename.startswith("face_") and filename.endswith(".png")]

            # Pobierz wyniki przetwarzania obrazów
            results = [future.result() for future in futures]

            # Przetwórz wyniki
            names_list = process_results(results)
    finally:
        end_time = time.time()  # Zakończ pomiar czasu
        execution_time = end_time - start_time
        print("Czas wykonania:", execution_time, "sekund")
        print(names_list)
    return names_list


get_names_list()