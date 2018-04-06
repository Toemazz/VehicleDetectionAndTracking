import cv2
from tqdm import tqdm
from VehicleDetectionAndTracking import VehicleDetectionAndTracking

# Set up video capture
video = cv2.VideoCapture('videos/video1_short.mp4')

# Get information about the videos
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create instances for vehicle detection
vdt = VehicleDetectionAndTracking(front=False)

for n in tqdm(range(n_frames)):
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
        cv2.imwrite('output/frame{}.png'.format(n+1), frame_out)
