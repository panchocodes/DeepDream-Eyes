# Turn directory of pngs into a mp4 video file.
import os
import sys
import cv2

directory = './video_file_path/'

video_dir = './video/'
video_name = video_dir+'video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 12.0, (1280, 720))

for filename in os.listdir(directory):
    name = directory + filename
    if (os.path.isfile(name)) and ".jpeg" in name:
        video.write(cv2.imread(name))

cv2.destroyAllWindows()
video.release()
