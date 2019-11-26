# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:04:35 2019

@author: Yousuf
"""


# import matplotlib.pyplot as plt
# imagePath = "20190329_145834.jpg"
# trainimage="IMG_0015.jpg"
import cv2
import sys
imagePath="/home/yousuf/Downloads/20191126_102436.jpg"
imagePath2="/home/yousuf/Downloads/20191126_102429.jpg"
image = cv2.imread(imagePath)
image2 = cv2.imread(imagePath2)

def comparison(imagePath,trainimage):
    import cv2
    import sys
    image = cv2.imread(imagePath)
    # print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        # print("[INFO] Object found. Saving locally.")
        # cv2.imwrite(str(w) + str(h) + 'xxx_roi_color_faces.jpg', roi_color)

    # status = cv2.imwrite('faces_detected.jpg', image)


    image2 = cv2.imread(trainimage)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    faceCascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces2 = faceCascade2.detectMultiScale(
        gray2,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces2:
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color2 = image2[y:y + h, x:x + w]
        # print("[INFO] Object found. Saving locally.")
        # cv2.imwrite(str(w) + str(h) + 'yyy_roi_color2_faces.jpg', roi_color2)

    # status = cv2.imwrite('faces_detected.jpg', image2)



    #print(roi_color2)

    import face_recognition
    import cv2
    import numpy as np
    import os
    import logging

    #IMAGES_PATH = './images'  # put your reference images in here
    CAMERA_DEVICE_ID = 0
    MAX_DISTANCE = 0.6# increase to make recognition less strict, decrease to make more strict

    # image0 = face_recognition.load_image_file("20190329_145834.jpg")
    # image0="660660_faces.jpg"
    # image1 = face_recognition.load_image_file("2013-08-10 15_59_17.jpg")
    # image1="397397_faces.jpg"

    #print(image)
    # imagex = cv2.imread(image0)
    rgb0 = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    # imagey = cv2.imread(image1)
    rgb1 = cv2.cvtColor(roi_color2, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb0,
            model='hog')
    boxes1=face_recognition.face_locations(rgb1,
            model='hog')
    document_encoding = face_recognition.face_encodings(rgb0,boxes)[0]
    selfie_encoding = face_recognition.face_encodings(rgb1,boxes1)[0]

    results = face_recognition.compare_faces([document_encoding], selfie_encoding)
    # print(results)
    distances = face_recognition.face_distance([document_encoding], selfie_encoding)
    print(distances)
    print(results[0])
    if distances<=0.57 or results[0]:
        result="True"
    else:
        result="False"
    print(result)
    return result

# def comparison(imagePath,trainimage):
#     from matplotlib import pyplot as plt
#     from mtcnn.mtcnn import MTCNN
#     # image = plt.imread(imagePath)
#     # detector = MTCNN()
#     # faces = detector.detect_faces(image)
#     # for face in faces:
#     #     print(face)
#     from matplotlib.patches import Rectangle
#     from numpy import asarray
#     from PIL import Image

#     def extract_face_from_image(image_path, required_size=(224, 224)):
#       # load image and detect faces
#         image = plt.imread(image_path)
#         detector = MTCNN()
#         faces = detector.detect_faces(image)

#         face_images = []

#         for face in faces:
#             # extract the bounding box from the requested face
#             x1, y1, width, height = face['box']
#             x2, y2 = x1 + width, y1 + height

#             # extract the face
#             face_boundary = image[y1:y2, x1:x2]

#             # resize pixels to the model size
#             face_image = Image.fromarray(face_boundary)
#             face_image = face_image.resize(required_size)
#             face_array = asarray(face_image)
#             face_images.append(face_array)

#         return face_images

#     extracted_face = extract_face_from_image(imagePath)

#     # Display the first face from the extracted faces
#     plt.imshow(extracted_face[0])
#     plt.show()
#     from keras_vggface.utils import preprocess_input
#     from keras_vggface.vggface import VGGFace
#     from scipy.spatial.distance import cosine
#     def get_model_scores(faces):
#         samples = asarray(faces, 'float32')

#         # prepare the data for the model
#         samples = preprocess_input(samples, version=2)

#         # create a vggface model object
#         model = VGGFace(model='resnet50',
#           include_top=False,
#           input_shape=(224, 224, 3),
#           pooling='avg')

#         # perform prediction
#         return model.predict(samples)

#     faces = [extract_face_from_image(image_path)
#              for image_path in [imagePath, trainimage]]

#     model_scores = get_model_scores(faces)
#     if cosine(model_scores[0], model_scores[1]) <= 0.4:
#       print("Faces Matched")

#     print(model_scores)



# comparison("/home/yousuf/Downloads/20191001_184719.jpg","/home/yousuf/Downloads/selfie.jpg")

#img1 = cv2.imread(imagePath,0)          # queryImage
# #img2 = cv2.imread(trainimage,0) # trainImage

# orb = cv2.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(roi_color,None)
# kp2, des2 = orb.detectAndCompute(roi_color2,None)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv2.drawMatches(roi_color,kp1,roi_color2,kp2,matches[:10],None, flags=2)
# plt.imshow(img3),plt.show()
# roi_color=cv2.resize(roi_color,(600,600))
# roi_color2=cv2.resize(roi_color2,(600,600))
# from skimage.measure import compare_ssim as ssim
# s = ssim(roi_color, roi_color2, multichannel=True)


