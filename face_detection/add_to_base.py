import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import psycopg2
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# Connect to the database - replace the SERVICE URI with the actual service URI
conn = psycopg2.connect(
    "postgres://avnadmin:AVNS_bfFc4wTNrJeCe8hQ3Yo@pg-29ab6d03-facesxd.a.aivencloud.com:12523/defaultdb?sslmode=require")


def add_to_base(employee,photo_path,direction):
    # Load a pre-trained model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Define image transformations
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])
    # Open the image
    img = Image.open(photo_path).convert('RGB')
    # Apply transformations
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    # Calculate the embeddings
    with torch.no_grad():
        embedding = resnet(img_tensor).cpu().numpy()
    # Convert the tensor to a numpy array
    embedding_np = embedding.squeeze()
    # Insert into the database
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures VALUES (DEFAULT,%s, %s,DEFAULT,%s)", (embedding_np.tolist(), employee, direction))
    conn.commit()
    # for filename in os.listdir("./gotowe"):
    #
    #     img = Image.open("./gotowe/" + filename).convert('RGB')
    #
    #     img_tensor = transform(img)
    #     # Add batch dimension
    #     img_tensor = img_tensor.unsqueeze(0).to(device)
    #     # Calculate the embeddings
    #     with torch.no_grad():
    #         embedding = resnet(img_tensor).cpu().numpy()
    #     # Convert the tensor to a numpy array
    #     embedding_np = embedding.squeeze()
    #     # Get the name of the person
    #     print(filename)
    #     name = input("Podaj nazwe osoby: ")
    #     # Insert into the database
    #     cur = conn.cursor()
    #     #up,down,left,right
    #     cur.execute("INSERT INTO pictures VALUES (DEFAULT,%s, %s,DEFAULT,%s)", (embedding_np.tolist(), name, 'up'))

    # Commit the changes to the database

