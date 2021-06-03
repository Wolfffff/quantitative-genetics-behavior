"""Base Analysis
"""

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from utils.utils import *
from itertools import groupby
from scipy.ndimage import gaussian_filter1d
import sklearn
import sleap.nn


hz = 10
bbs = "/Volumes/T5/sleap_data/f1_expmt/expmt_files/f1s_set1_plates123.csv"
md = "/Users/wolf/Dropbox (Princeton)/f1_experiment/data/F1s_set1/190717_F1_camera1.csv"

bound_boxes = pd.read_csv(bbs, header=None).to_numpy()
metadata = pd.read_csv(md)
metadata.Empty = metadata.Empty == 'T'

f = h5py.File('/Users/wolf/Google Drive/Colab Notebooks/sleap_data/predictions/merged_1-20.slp', 'r')
# f = h5py.File('/Users/wolf/git/QGB/qgb_python/data/raw_data/20200603_Working_Proj.analysis.h5','r')
instances = f['instances'][:]
pred_points = f['pred_points'][:]
frames = f['frames'][:]
instance_ids = pd.DataFrame([frames['instance_id_start'], frames['instance_id_end']]).transpose()
