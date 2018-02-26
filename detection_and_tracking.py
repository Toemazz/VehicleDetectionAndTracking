import sys
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

from utilities.vehicle_detector import VehicleDetector
from utilities.vehicle_tracker import VehicleTracker
from utilities.bounding_box import *


class VehicleDetectionAndTracking:
    def __init__(self, min_conf=0.7, max_age=2, max_hits=8, display=False):
        # Initialize constants
        self.frame_count = 0
        self.max_age = max_age                   # no. of consecutive unmatched detection before a track is deleted
        self.min_hits = max_hits                 # no. of consecutive matches needed to establish a track
        self.tracker_list = []
        self.track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
        self.display = display

        # Set up 'Vehicle Detector'
        self.detector = VehicleDetector(kitti=False, min_conf=min_conf)

    # Method: Used to match detections to trackers
    @staticmethod
    def match_detections_to_trackers(trackers, detections, min_iou=0.25):
        # Initialize 'iou_matrix'
        iou_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)

        # Populate 'iou_matrix'
        for t, tracker in enumerate(trackers):
            for d, detection in enumerate(detections):
                iou_matrix[t, d] = box_iou_ratio(tracker, detection)

        # Produce matches by using the Hungarian algorithm to maximize the sum of IOU
        matched_index = linear_assignment(-iou_matrix)

        # Populate 'unmatched_trackers'
        unmatched_trackers = []
        for t in np.arange(len(trackers)):
            if t not in matched_index[:, 0]:
                unmatched_trackers.append(t)

        # Populate 'unmatched_detections'
        unmatched_detections = []
        for d in np.arange(len(detections)):
            if d not in matched_index[:, 1]:
                unmatched_detections.append(d)

        # Populate 'matches'
        matches = []
        for m in matched_index:
            # Create tracker if IOU is greater than 'min_iou'
            if iou_matrix[m[0], m[1]] > min_iou:
                matches.append(m.reshape(1, 2))
            else:
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])

        if matches:
            # Concatenate arrays on the same axis
            matches = np.concatenate(matches, axis=0)
        else:
            matches = np.empty((0, 2), dtype=int)

        # Return matches, unmatched detection and unmatched trackers
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    # Method: Used as a 'pipeline' function for detection and tracking
    def pipeline(self, image):
        self.frame_count += 1

        # Get bounding boxes for located vehicles
        det_boxes = self.detector.get_bounding_box_locations(image)

        # Add detected bounding boxes to the image
        if self.display:
            for i in np.arange(len(det_boxes)):
                image1 = draw_box_label(image, det_boxes[i], colour=(255, 0, 0))
                plt.imshow(image1)
            plt.show()

        # Get list of tracker bounding boxes
        trk_boxes = []
        if self.tracker_list:
            for tracker in self.tracker_list:
                trk_boxes.append(tracker.box)

        # Match detected vehicles to trackers
        matched, unmatched_dets, unmatched_trks = self.match_detections_to_trackers(trk_boxes, det_boxes)

        # Deal with matched detections
        if len(matched) > 0:
            for trk_idx, det_idx in matched:
                z = det_boxes[det_idx]
                z = np.expand_dims(z, axis=0).T
                temp_trk = self.tracker_list[trk_idx]
                temp_trk.predict_and_update(z)
                xx = temp_trk.x_state.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                trk_boxes[trk_idx] = xx
                temp_trk.box = xx
                temp_trk.num_hits += 1

        # Deal with unmatched detections
        if len(unmatched_dets) > 0:
            for i in unmatched_dets:
                z = det_boxes[i]
                z = np.expand_dims(z, axis=0).T
                temp_trk = VehicleTracker()  # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                temp_trk.x_state = x
                temp_trk.predict()
                xx = temp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                temp_trk.box = xx
                temp_trk.id = self.track_id_list.popleft()  # assign an ID for the tracker
                self.tracker_list.append(temp_trk)
                trk_boxes.append(xx)

        # Deal with unmatched tracks
        if len(unmatched_trks) > 0:
            for i in unmatched_trks:
                temp_trk = self.tracker_list[i]
                temp_trk.num_unmatched += 1
                temp_trk.predict()
                xx = temp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                temp_trk.box = xx
                trk_boxes[i] = xx

        # Populate the list of trackers to be displayed on the image
        good_tracker_list = []
        for tracker in self.tracker_list:
            if tracker.num_hits >= self.min_hits and tracker.num_unmatched <= self.max_age:
                good_tracker_list.append(tracker)
                tracker_bb = tracker.box

                # Draw bounding box on the image
                image = draw_box_label(image, tracker_bb)

        # Find list of trackers to be deleted
        deleted_trackers = filter(lambda x: x.num_unmatched > self.max_age, self.tracker_list)

        for tracker in deleted_trackers:
            self.track_id_list.append(tracker.id)

        # Update list of active trackers
        self.tracker_list = [x for x in self.tracker_list if x.num_unmatched <= self.max_age]

        return image

    # Method: Used to end VideoFileClip processes
    @staticmethod
    def close_clip(clip):
        try:
            clip.reader.close()
            del clip.reader

            if clip.audio is not None:
                clip.audio.reader.close_proc()
                del clip.audio

            del clip
        except Exception:
            sys.exc_clear()


if __name__ == "__main__":
    vdt = VehicleDetectionAndTracking(min_conf=0.5, max_age=2, max_hits=8)
    output = 'video1_short_test.mp4'
    input_vid = VideoFileClip('video1_short.mp4')
    output_vid = input_vid.fl_image(vdt.pipeline)
    output_vid.write_videofile(output, audio=False)
    vdt.close_clip(output_vid)
