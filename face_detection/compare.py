import numpy as np
from imgbeddings import imgbeddings
from PIL import Image as Image1
from IPython.display import Image, display as IPImage,display
import psycopg2
import os
import ast

conn = psycopg2.connect("postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require")
#tu wrzucić z poprzedniego modelu efekt koncowy
file_name = "./compare/1.png"  # replace <INSERT YOUR FACE FILE NAME> with the path to your image
# opening the image
img =Image1.open(file_name)
# loading the `imgbeddings`
ibed = imgbeddings()
# calculating the embeddings
embedding = ibed.to_embeddings(img)


cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
for row in rows:
    embedded_image = np.array(ast.literal_eval(row[1]))  # Osadzenie z bazy danych
    norm_distance = np.linalg.norm(embedded_image - embedding[0])  # Odległość między osadzeniami
    print("Odległość między osadzeniami:", norm_distance)
    print(row[2])
cur.close()