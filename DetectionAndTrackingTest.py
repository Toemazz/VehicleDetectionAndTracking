import cv2
import numpy as np

from tqdm import tqdm
from DetectionAndTracking import VehicleDetectionAndTracking

# Set up video capture
left_video = cv2.VideoCapture('')
right_video = cv2.VideoCapture('')

# Get information about the videos
n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)), int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
fps = int(left_video.get(cv2.CAP_PROP_FPS))

# Create instances for vehicle detection
left_vdt = VehicleDetectionAndTracking(left=True)
right_vdt = VehicleDetectionAndTracking(left=False)

for n in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    ok, left_in = left_video.read()
    _, right_in = right_video.read()

    if ok:
        # Process images
        left_out = left_vdt.pipeline(left_in)
        right_out = left_vdt.pipeline(right_in)

        # Check if any vehicles were detected in the 'danger zones'
        left_vehicle_detected = left_vdt.vehicle_detected
        right_vehicle_detected = right_vdt.vehicle_detected

        # Horizontally concatenate images and resize
        out = np.hstack([left_out, right_out])
        out = cv2.resize(out, (0, 0), fx=0.5, fy=0.5)

        # Add 'SAFE' to image when no warnings were issued
        if (left_vehicle_detected == False) and (right_vehicle_detected == False):
            dims = out.shape[:2]
            cv2.putText(out, 'SAFE', (int(dims[1]/2)-40, 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        if n < 10:
            cv2.imwrite('{}.png'.format(n), out)
