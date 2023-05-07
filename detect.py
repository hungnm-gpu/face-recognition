from insightface_model import model_insightface
from detect_face import result,result_img,result_main
import os
import time
import cv2
import faiss
import numpy as np
import codecs
import json



rec, retrain, detector = model_insightface()
index = faiss.IndexFlatL2(512)

dataset = np.load('data/dataset.data.npy')
with codecs.open('data/labels.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)
    f.close()
with codecs.open('data/code_dict.json', 'r', encoding='utf-8') as f:
    code_dict = json.load(f)
    f.close()
with codecs.open('data/name_dict.json', 'r', encoding='utf-8') as f:
    name_dict = json.load(f)
    f.close()
index.reset()
index.add(dataset)


img = cv2.imread('12.png')
img = result_main(img)
cv2.imshow("img.img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# video = cv2.VideoCapture(0)
# size = (960, 540)

# # Below VideoWriter object will create
# # a frame of above defined The output 
# # is stored in 'filename.avi' file.
# result = cv2.VideoWriter('filename.avi', 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)
# while True:
#     _,frame = video.read()
#     img = result_main(rec, index, labels, dataset, code_dict, name_dict, detector, frame)
#     img_save = cv2.resize(img,size)
#     result.write(img_save)
#     cv2.imshow("img.img", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video.release()
# result.release()
# cv2.destroyAllWindows()

