# VehicleDetectionAndTracking

## Description
This project uses the Tensorflow object detection model as a base. This is a CNN which is pre-trained on a number of different datasets. The model is pre-trained on the Kitti and COCO datasets which can be used for Vehicle and VRU classification (as well as other objects). 

The object detection model was minimized to return the pixel coordinates of the bounding boxes for detected vehicles. These coordinates were then fed into a Kalman filter for 'next state estimation' and tracking.

## Datasets
### Kitti Dataset Classes
- Cars
- Pedestrians

### COCO Dataset Classes
- Cars
- Trucks
- Buses
- Trains
- Motorcyclists
- Cyclists
- Pedestrians
