from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from insightface_model import model_insightface
from detect_face import result
import os
import time
import cv2

print(cv2.__version__)