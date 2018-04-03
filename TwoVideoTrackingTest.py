import cv2
import numpy as np
from random import randint


def add_random_points_to_image(dimensions, n_points, colour):
    copy = np.zeros(dimensions, dtype=np.uint8)
    copy[:] = 255
    points = []

    for i in range(n_points):
        rand_x = randint(0, dimensions[1])
        rand_y = randint(0, dimensions[0])

        cv2.circle(copy, (rand_x, rand_y), 4, colour, -1)
        points.append((rand_y, rand_x))

    return copy, points


# Initialize parameters
dims = (270, 480, 3)
x_distance = 50
p_distance = 5
num_points = 100

# Add random points to image
left, l_pts = add_random_points_to_image(dims, num_points, (255, 0, 0))
right, r_pts = add_random_points_to_image(dims, num_points, (0, 255, 0))
print(l_pts)
# Find points to be removed
l_pts_del, r_pts_del = [], []
for i in range(num_points):
    if l_pts[i][1] >= dims[1]-x_distance:
        l_pts_del.append(l_pts[i])

    if r_pts[i][1] <= x_distance:
        r_pts_del.append(r_pts[i])

# Create output image
output = np.hstack((left, right))

# Find points with similar y-coordinates
for l_pt in l_pts_del:
    for r_pt in r_pts_del:
        if abs(l_pt[0]-r_pt[0]) <= p_distance:
            cv2.line(output, (l_pt[1], l_pt[0]), (r_pt[1]+dims[1], r_pt[0]), (0, 0, 0), 2)
            print('MATCHED {} and {}'.format(l_pt, r_pt))

# Draw horizontal lines
cv2.line(output, (dims[1]-x_distance, 0), (dims[1]-x_distance, 2*dims[0]), (0, 0, 0), 2)
cv2.line(output, (dims[1]+x_distance, 0), (dims[1]+x_distance, 2*dims[0]), (0, 0, 0), 2)

# Display output
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
