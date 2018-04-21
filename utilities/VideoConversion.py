import os
import cv2
from tqdm import tqdm as tq
import subprocess


# Method: Used to convert a video into frames
def video_to_frames(video_path, output_path):
    """
    :param video_path: Path to original video
    :param output_path: Path to output frames
    """
    count, frame_num = 1, 1

    # Get video
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tq(range(1, total_frames+1)):
        # Read frame
        ok, img = video.read()

        if ok:
            # Save frame
            frame_name = 'frame{}.png'.format(frame_num)
            frame_path = os.path.join(output_path, frame_name)
            cv2.imwrite(frame_path, img)
            frame_num += 1
            count += 1
            # if 1200 <= count < 2550:
            #     # Save frame
            #     frame_name = 'frame{}.png'.format(frame_num)
            #     frame_path = os.path.join(output_path, frame_name)
            #     cv2.imwrite(frame_path, img)
            #     frame_num += 1
            # count += 1


# Method: used to create a video from a list of frames
def frames_to_video(input_dir, output_path, frame_format='frame', file_type='.png', codec='libx264', fps=30):
    """
    :param input_dir: Path to image directory
    :param output_path: Path to video
    :param frame_format: Frame Format
    :param file_type: Image file extension
    :param codec: Video codec
    :param fps: Frames/second
    """
    input_image_format = '{}/{}%d{}'.format(input_dir, frame_format, file_type)
    subprocess.call('ffmpeg -y -r {} -i {} -vcodec {} {}'.format(fps, input_image_format, codec, output_path))


# video_to_frames('C:/PythonProjects/VehicleDetectionAndTracking/front_left_vlc_2.mp4',
#                 'C:/PythonProjects/VehicleDetectionAndTracking/output/front_left_2_out/')

frames_to_video('C:/PythonProjects/VehicleDetectionAndTracking/output/front_both_1_out/',
                'C:/PythonProjects/VehicleDetectionAndTracking/videos/front_both_1_out.mp4', fps=30)
