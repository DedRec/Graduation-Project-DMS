import pandas as pd
import numpy as np
import scipy.io as sio
from PIL import Image
import os

inputd = pd.read_csv("files.txt")
outputd = pd.DataFrame()
datapath = "./"

def image_to_array(path):
    img_path = Image.open(os.path.join(datapath, path + '.jpg'))
    img_path = img_path.convert('RGB')
    img_array = np.asarray(img_path)
    return img_array

def extract_features(path):
    mat = sio.loadmat(path)
    pre_pose_params = mat['Pose_Para'][0]
    pitch, yaw, roll = pre_pose_params[:3]
    return yaw, pitch, roll

def extractmin_max(path):
    mat = sio.loadmat(path)
    pt2d = mat['pt2d']
    x_min = min(pt2d[0, :])
    y_min = min(pt2d[1, :])
    x_max = max(pt2d[0, :])
    y_max = max(pt2d[1, :])

    k = np.random.random_sample() * 0.2 + 0.2
    x_min -= 0.6 * k * abs(x_max - x_min)
    y_min -= 2 * k * abs(y_max - y_min)
    x_max += 0.6 * k * abs(x_max - x_min)
    y_max += 0.6 * k * abs(y_max - y_min)
    return x_min, y_min, x_max, y_max

outputd["image"] = inputd["files"]
outputd[["yaw", "pitch", "roll"]] = inputd["files"].apply(extract_features).tolist()
outputd[["x_min", "y_min", "x_max", "y_max"]] = inputd["files"].apply(extractmin_max).tolist()

outputd.to_pickle("file.pkl")
print(outputd.head())
