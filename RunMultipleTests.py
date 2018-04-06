import ntpath
from VehicleDetectionAndTracking import VehicleDetectionAndTracking
from moviepy.editor import VideoFileClip


input_file_name = 'videos/video1_short.mp4'
output_dir = 'output'

# Confidence Level
for i in range(60, 76, 5):
    # Number of consecutive unmatched detections
    for j in range(1, 3, 1):
        # Number of consecutive matches needed
        for k in range(8, 13, 2):
            i_conf = i / 100
            # -------------------------------------------------
            vdt = VehicleDetectionAndTracking(min_conf=i_conf, max_age=j, max_hits=k)
            output = '{}/{}_{}_{}_{}.mp4'.format(output_dir, ntpath.basename(input_file_name)[:-4],
                                                 str(i).zfill(2), str(j).zfill(2), str(k).zfill(2))

            input_vid = VideoFileClip(input_file_name)
            output_vid = input_vid.fl_image(vdt.pipeline).resize(width=480)
            output_vid.write_videofile(output, audio=False)
            vdt.close_clip(output_vid)
            # -------------------------------------------------
