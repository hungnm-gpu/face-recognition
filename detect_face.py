import cv2
import faiss
import numpy as np
import codecs
import json
import time
from insightface_model import model_insightface
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def result(rec, index, labels, dataset, code_dict, name_dict, detector, path):
    # last_name = 'Unknown'
    img = cv2.imread(path)
    _, kpss1 = detector.autodetect(img, max_num=2)
    message = ''
    type_result = ''
    if len(kpss1) == 0:
        message = "Không phát hiện được khuôn mặt"
        type_result = 1
    elif len(kpss1) > 1:
        message = "Vui lòng tải ảnh chỉ có 1 khuôn mặt"
    else:
        feat1 = rec.get(img,kpss1[0])
        queries = np.array([feat1])
        _, I = index.search(queries, 2)  # actual search
        # name = 'Unknown'
        feat2 = dataset[I[0][0]]
        sim = rec.compute_sim(feat1, feat2)
        index_str = str(I[0][0])
        
        if labels[index_str] not in code_dict:
            # face_name = 'NO NAME'
            message = 'Không tìm thấy'
        elif sim < 0.4:
            # face_name = 'Unknown'
            message = 'Không nhận diện được sinh viên trong ảnh'
        elif 0.4 <= sim < 0.45:
            # face_name = 'LIKELY: {} - {}'.format(code_dict[labels[index_str]], name_dict[labels[index_str]])
            message = f'Người trong ảnh có vẻ giống {code_dict[labels[index_str]]}, {name_dict[labels[index_str]]}'
            # if face_name != last_name:
            #     print('{:0.2f} {}'.format(sim, face_name))
        else:
            # face_name = '{} - {}'.format(code_dict[labels[index_str]], name_dict[labels[index_str]])
            message = f'Người trong ảnh là {code_dict[labels[index_str]]}, {name_dict[labels[index_str]]}'
            # if face_name != last_name:
            #     print('{:0.2f} {}'.format(sim, face_name))
        # last_name = face_name
        # name = '{:0.2f} {}'.format(sim, face_name)

    return message, type_result
def result_img(rec, index, labels, dataset, code_dict, name_dict, detector, img):
    # last_name = 'Unknown'
    _, kpss1 = detector.autodetect(img, max_num=2)
    message = ''
    type_result = ''
    if len(kpss1) == 0:
        message = "Không phát hiện được khuôn mặt"
        type_result = 1
    elif len(kpss1) > 1:
        message = "Vui lòng tải ảnh chỉ có 1 khuôn mặt"
    else:
        feat1 = rec.get(img,kpss1[0])
        queries = np.array([feat1])
        _, I = index.search(queries, 2)  # actual search
        # name = 'Unknown'
        feat2 = dataset[I[0][0]]
        sim = rec.compute_sim(feat1, feat2)
        index_str = str(I[0][0])
        
        if labels[index_str] not in code_dict:
            # face_name = 'NO NAME'
            message = 'unknown'
        simm = "{:.2f}".format(sim)
        message = f'{code_dict[labels[index_str]]}, {name_dict[labels[index_str]]} {simm}'

    return message, type_result

def result_main(rec, index, labels, dataset, code_dict, name_dict, detector, img):
    # last_name = 'Unknown'
    boxs, kpss = detector.autodetect(img, max_num=20) 
    img_draw = img.copy()   
    message = ''
    type_result = ''
    tt = 0
    for i in range(len(kpss)):
        t1 = time.time()
        x1,y1,x2,y2,z = boxs[i]
        feat1 = rec.get(img,kpss[i])

        tt += time.time() - t1
        print(f"time{i}: ", tt)

        queries = np.array([feat1])
        _, I = index.search(queries, 2)  # actual search
        # name = 'Unknown'
        feat2 = dataset[I[0][0]]

        sim = rec.compute_sim(feat1, feat2)
        index_str = str(I[0][0])
        
        if sim > 0.3:
            simm = "{:.2f}".format(sim)
            # message1 = f'{code_dict[labels[index_str]]}, {name_dict[labels[index_str]]} {simm}'
            message = f'{code_dict[labels[index_str]]} {simm}'
            # print(message1)
            plot_one_box([x1,y1,x2,y2], img_draw, (0, 255,0), label=message, line_thickness=3)
        else:
            plot_one_box([x1,y1,x2,y2], img_draw, (0, 0,255), label='unknown', line_thickness=3)


    return img_draw


