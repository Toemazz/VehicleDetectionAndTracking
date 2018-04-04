from moviepy.editor import VideoFileClip
from DetectionAndTracking import VehicleDetectionAndTracking

vdt = VehicleDetectionAndTracking(left=False)
output = 'video1_short_test.mp4'
input_vid = VideoFileClip('videos/video1_short.mp4')
output_vid = input_vid.fl_image(vdt.pipeline)
output_vid.write_videofile(output, threads=4, audio=False)
vdt.close_clip(output_vid)
