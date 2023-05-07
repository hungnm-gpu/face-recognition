from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from insightface_model import model_insightface
from detect_face import result,result_img,result_main
import os
import time
import cv2
import faiss
import numpy as np
import codecs
import json
import onnxruntime
import uvicorn

    
from typing import List
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

IMAGEDIR = "images/"

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


app = FastAPI()


templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")
 
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.post("/upload-files")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):


    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = result_main(rec, index, labels, dataset, code_dict, name_dict, detector, img)
        cv2.imwrite(f"{IMAGEDIR}{file.filename}", img)
    show = [file.filename for file in files]
 
    #return {"Result": "OK", "filenames": [file.filename for file in files]}
    return templates.TemplateResponse("index.html", {"request": request, "show": show})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)