import glob
import ntpath

from DetectionAndTracking import VehicleDetectionAndTracking
from utilities.video_conversion import *


def delete_files_in_dir(dir_path):
    files = glob.glob(os.path.join(dir_path, '.*'))
    for f in files:
        os.remove(f)


input_file_name = 'videos/video1_short.mp4'
output_dir = 'output'
images_dir = 'images'
images_out_dir = 'images_out'

video_to_frames(input_file_name, images_dir)
image_names = os.listdir(images_dir)

# Confidence Level
for i in range(50, 76, 5):
    # Number of consecutive unmatched detections
    for j in range(1, 2, 1):
        # Number of consecutive matches needed
        # for k in range(8, 11, 2):
        k = 8
        i_conf = i / 100
        # -------------------------------------------------
        vdt = VehicleDetectionAndTracking(min_conf=i_conf, max_age=j, max_hits=k)
        output = '{}/{}_{}_{}_{}.mp4'.format(output_dir, ntpath.basename(input_file_name)[:-4],
                                             str(i).zfill(2), str(j).zfill(2), str(k).zfill(2))

        for i in tq(range(len(image_names)), desc=output):
            file_path = os.path.join(images_dir, image_names[i])
            img = cv2.imread(file_path)
            img_out = vdt.pipeline(img)

            file_path_out = os.path.join(images_out_dir, image_names[i])
            cv2.imwrite(file_path_out, img_out)

        frames_to_video(images_out_dir, output, fps=25)
        delete_files_in_dir(images_out_dir)
