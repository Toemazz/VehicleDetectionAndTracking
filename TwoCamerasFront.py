import cv2
import numpy as np
from tqdm import tqdm

from VehicleDetectionAndTrackingProject import VehicleDetectionAndTrackingProject

# Set up video capture
left_video = cv2.VideoCapture('videos/front_left_1.mp4')
right_video = cv2.VideoCapture('videos/front_right_1.mp4')

# Get information about the videos
n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)), int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
fps = int(left_video.get(cv2.CAP_PROP_FPS))

# Create instances for vehicle detection
left_vdt = VehicleDetectionAndTrackingProject(left=True)
right_vdt = VehicleDetectionAndTrackingProject(left=False)

for n in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    ok, left_in = left_video.read()
    _, right_in = right_video.read()

    if ok:
        # Process images
        left_out = left_vdt.pipeline(left_in)
        right_out = right_vdt.pipeline(right_in)

        # Get warnings
        left_warning = left_vdt.warning
        right_warning = right_vdt.warning

        # Horizontally concatenate images and resize
        out = np.hstack([left_out, right_out])
        out = cv2.resize(out, (0, 0), fx=0.5, fy=0.5)

        # Add 'SAFE' to image when no warnings were issued
        if (not left_warning) and (not right_warning):
            dims = out.shape[:2]
            cv2.putText(out, 'SAFE', (int(dims[1]/2)-40, 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Save frame
        cv2.imwrite('output/front_both_1_out/frame{}.png'.format(n+1), out)
