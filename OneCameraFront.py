import cv2
import numpy as np
from tqdm import tqdm

from VehicleDetectionAndTrackingProject import VehicleDetectionAndTrackingProject

# Set up video capture
video = cv2.VideoCapture('videos/front_right_2.mp4')

# Get information about the videos
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create instances for vehicle detection
vdt = VehicleDetectionAndTrackingProject(front=True, left=False)

for n in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    ok, frame = video.read()

    if ok:
        # Process images
        out = vdt.pipeline(frame)

        # Get warnings
        warning = vdt.warning

        # Add 'SAFE' to image when no warnings were issued
        if not warning:
            dims = out.shape[:2]
            cv2.putText(out, 'SAFE', (int(dims[1]/2)-80, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Save frame
        cv2.imwrite('output/front_right_2_out/frame{}.png'.format(n+1), out)
