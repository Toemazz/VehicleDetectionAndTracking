from moviepy.editor import VideoFileClip
from VehicleDetectionAndTracking import VehicleDetectionAndTracking

vdt = VehicleDetectionAndTracking(front=False)
output = 'right_60_01_oncoming_02_test_rear.mp4'
input_vid = VideoFileClip('videos/right_60_01_oncoming_02.mp4')
output_vid = input_vid.fl_image(vdt.pipeline)
output_vid.write_videofile(output, threads=4, audio=False)
vdt.close_clip(output_vid)
