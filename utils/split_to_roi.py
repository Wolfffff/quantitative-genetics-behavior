import pandas as pd
import ffmpeg
import sys
import os

vid = sys.argv[1]
metadata = sys.argv[2]
output_directory = sys.argv[3]

print(vid)
print(metadata)
print(output_directory)

roi_corners_f1s_set1_cam1 = pd.read_csv(metadata, header=None)
rois = roi_corners_f1s_set1_cam1.to_numpy()

def split_video(video_name_without_suffix, vid_type_suffix, bb, index, output_dir,input_prefix=""):
    stream = ffmpeg.input(os.getcwd() +"/" +input_prefix + video_name_without_suffix + vid_type_suffix)
    stream = ffmpeg.crop(stream, x=bb[0], y=bb[1], width=bb[2], height=bb[3])
    stream = ffmpeg.output(stream, output_dir + video_name_without_suffix + "_" + str(index) + vid_type_suffix, preset='fast',codec="h264_nvenc")
    ffmpeg.run(stream)

for i in range(0, rois.shape[0]):
    split_video(video_name_without_suffix=vid, vid_type_suffix=".mp4", bb=rois[i, :], index=i,
                output_dir=output_directory)
# /tigress/swwolf/qgb/indiv_sleap_data/split_videos/f1s_set1_plates123/
