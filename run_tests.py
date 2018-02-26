from detection_and_tracking import VehicleDetectionAndTracking
from moviepy.editor import VideoFileClip


input_file_name = 'videos/video2.mp4'
output_dir = 'output'

# Confidence Level
for i in range(60, 81, 10):
    # Number of consecutive unmatched detections
    for j in range(1, 5, 1):
        # Number of consecutive matches needed
        for k in range(6, 11, 2):
            i_conf = i / 100
            # -------------------------------------------------
            vdt = VehicleDetectionAndTracking(min_conf=i_conf, max_age=j, max_hits=k)
            output = '{}/{}_{}_{}_{}.mp4'.format(output_dir, input_file_name[:-4],
                                                 str(i).zfill(2), str(j).zfill(2), str(k).zfill(2))
            input_vid = VideoFileClip(input_file_name)
            output_vid = input_vid.fl_image(vdt.pipeline)
            output_vid.write_videofile(output, audio=False)
            vdt.close_clip(output_vid)
            # -------------------------------------------------
