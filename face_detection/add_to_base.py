import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

# connecting to the database - replace the SERVICE URI with the service URI
conn = psycopg2.connect("postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require")

for filename in os.listdir("./gotowe"):
    # opening the image
    img = Image.open("./gotowe/" + filename)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    name = input("Podaj nazwe osoby: ")
    cur.execute("INSERT INTO pictures values (%s,%s,%s)", (filename, embedding[0].tolist(),name))
    print(filename)
conn.commit()