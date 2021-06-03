import sleap
import sleap.info.write_tracking_h5
import os
from tqdm import tqdm

os.chdir("/Users/wolf/Google Drive/Colab Notebooks/sleap_data/predictions/f1s_set1_plates123")

base_filename = "2020-05-31_f1s_77_118_plates1_2_3_20190717_182307.mp4_"
filename_suffix = ".slp"

base_labels = sleap.Labels.load_file("2020-05-31_f1s_77_118_plates1_2_3_20190717_182307.mp4_1.slp")

for i in tqdm(range(2,1001)):
    new_labels = sleap.Labels.load_file(base_filename + str(i) + filename_suffix)
    base_labels = sleap.Labels(base_labels.labeled_frames + new_labels.labeled_frames)



sleap.Labels.save_file(base_labels, "merged_1000_done.slp")
