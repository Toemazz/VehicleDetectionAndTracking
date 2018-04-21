import sys
import numpy as np
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

from utilities.VehicleDetector import VehicleDetector
from utilities.VehicleTracker import VehicleTracker
from utilities.BoundingBox import *


class VehicleDetectionAndTrackingProject:
    def __init__(self, min_conf=0.6, max_age=4, max_hits=10, front=True, left=False):
        # Initialize constants
        self.max_age = max_age                   # no. of consecutive unmatched detection before a track is deleted
        self.min_hits = max_hits                 # no. of consecutive matches needed to establish a track
        self.tracker_list = []
        self.track_id_list = deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
        self.left = left
        self.front = front
        self.vehicle_detected = False
        self.count = 0

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
        dims = image.shape[:2]
        self.count += 1

        # Get bounding boxes for located vehicles
        det_boxes = self.detector.get_bounding_box_locations(image)

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
        warning_count = 0
        area_count = 0

        for tracker in self.tracker_list:
            if tracker.num_hits >= self.min_hits and tracker.num_unmatched <= self.max_age:
                good_tracker_list.append(tracker)
                tracker_bb = tracker.box

                # Draw bounding box on the image
                image = draw_box_label(image, tracker_bb)

                if self.front:
                    center = (int(np.average([tracker_bb[0], tracker_bb[2]])),
                              int(np.average([tracker_bb[1], tracker_bb[3]])))

                    if self.left:
                        if center[1] <= dims[1] // 2:
                            cv2.putText(image, 'WARNING', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 2,
                                        cv2.LINE_AA)
                            warning_count += 1
                    else:
                        if center[1] >= dims[1] // 2:
                            cv2.putText(image, 'WARNING', (dims[1]-300, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 2,
                                        cv2.LINE_AA)
                            warning_count += 1
                else:
                    h = tracker_bb[2] - tracker_bb[0]
                    w = tracker_bb[3] - tracker_bb[1]
                    bb_area = h * w
                    bb_area_percent = 100 * (bb_area/(dims[0]*dims[1]))

                    if bb_area_percent >= 2:
                        cv2.putText(image, 'WARNING', (int(dims[1]/2)-120, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        area_count += 1

        # Find list of trackers to be deleted
        deleted_trackers = filter(lambda x: x.num_unmatched > self.max_age, self.tracker_list)

        for tracker in deleted_trackers:
            self.track_id_list.append(tracker.id)

        # Update list of active trackers
        self.tracker_list = [x for x in self.tracker_list if x.num_unmatched <= self.max_age]

        # True if vehicle was detected in a 'danger zone'
        if self.front:
            self.warning = True if warning_count > 0 else False
        else:
            self.warning = True if area_count > 0 else False

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
