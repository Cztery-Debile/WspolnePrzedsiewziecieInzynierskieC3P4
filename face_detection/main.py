# importing the cv2 library
import cv2
import os
import random

import numpy

# loading the haar case algorithm file into alg variable
alg = "./haarcascade_frontalface_default.xml"
# passing the algorithm to OpenCV
haar_cascade = cv2.CascadeClassifier(alg)
# loading the image path into file_name variable - replace <INSERT YOUR IMAGE NAME HERE> with the path to your image
directory ="./test"
random.seed(1455512)
# iterate over files in
# that directory
for filename in os.scandir(directory):
    file_name = ""
    # reading the image
    print(str(filename.name))
    img = cv2.imread(os.path.join(directory,filename.name), 0)
    # creating a black and white version of the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # detecting the faces
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.05, minNeighbors=3, minSize=(10, 10)
    )

    # for each face detected
    for x, y, w, h in faces:
        # crop the image to select only the face

        print ('znalazlo')

        xd = random.randint(3, 10000)
        print(x)
        random.seed(random.randint(2,200))
        cropped_image = img[y : y + h, x : x + w]
        # loading the target image path into target_file_name variable  - replace <INSERT YOUR TARGET IMAGE NAME HERE> with the path to your target image
        target_file_name = './gotowe/' + str(xd) + '.jpg'
        print(type(cropped_image))
        #
        # if type(cropped_image) is numpy.ndarray:
        #     print("Nie udało się wczytać obrazu.")
        # else:
        cv2.imwrite(
        target_file_name,
        cropped_image,
        )
