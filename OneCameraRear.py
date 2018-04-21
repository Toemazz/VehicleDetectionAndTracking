import cv2
from tqdm import tqdm
from VehicleDetectionAndTrackingProject import VehicleDetectionAndTrackingProject

# Set up video capture
video = cv2.VideoCapture('videos/rear_right_2.mp4')

# Get information about the videos
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create instances for vehicle detection
vdt = VehicleDetectionAndTrackingProject(front=False)

for _ in tqdm(range(n_frames)):
    # Grab the frames from their respective video streams
    ok, frame_in = video.read()

    if ok:
        # Process images
        frame_out = vdt.pipeline(frame_in)

        # Get warning
        frame_warning = vdt.warning

        # Add 'SAFE' to image when no warnings were issued
        if not frame_warning:
            dims = frame_out.shape[:2]
            cv2.putText(frame_out, 'SAFE', (int(dims[1]/2)-80, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 0), 2,
                        cv2.LINE_AA)

        # Save frame
        cv2.imwrite('output/rear_right_2_out_test/frame{}.png'.format(vdt.count), frame_out)
