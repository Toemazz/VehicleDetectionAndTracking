import six
import collections
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

standard_colors = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


# Method: Used to add a bounding box to an image (numpy array)
def draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, point, color='red', thickness=4, display_str_list=(),
                                     use_normalized_coordinates=True):
    """
    :param image: A numpy array with shape [height, width, 3]
    :param ymin: ymin of bounding box
    :param xmin: xmin of bounding box
    :param ymax: ymax of bounding box
    :param xmax: xmax of bounding box
    :param color: Color to draw bounding box
    :param thickness: Line thickness
    :param display_str_list: List of strings to display in box (each to be shown on its own line)
    :param use_normalized_coordinates: If True, treat coordinates ymin, xmin, ymax, xmax as relative to the image.
        Otherwise treat coordinates as absolute
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, point, color, thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


# Method: Used to add a bounding box to an image
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, center, color='red', thickness=4, display_str_list=(),
                               use_normalized_coordinates=True):
    """
    :param image: A numpy array with shape [height, width, 3]
    :param ymin: ymin of bounding box
    :param xmin: xmin of bounding box
    :param ymax: ymax of bounding box
    :param xmax: xmax of bounding box
    :param color: Color to draw bounding box
    :param thickness: Line thickness
    :param display_str_list: List of strings to display in box (each to be shown on its own line)
    :param use_normalized_coordinates: If True, treat coordinates ymin, xmin, ymax, xmax as relative to the image.
        Otherwise treat coordinates as absolute
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding box exceeds the top of the image,
    # stack the strings below the bounding box instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
        text_bottom -= text_height - 2 * margin


# Method: Used to draw keypoints on an image (numpy array)
def draw_keypoints_on_image_array(image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
    """
    :param image: A numpy array with shape [height, width, 3]
    :param keypoints: A numpy array with shape [num_keypoints, 2]
    :param color: Color to draw the keypoints with
    :param radius: Keypoint radius
    :param use_normalized_coordinates: If True, treat keypoint values as relative to the image.
        Otherwise treat them as absolute
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius, use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


# Method: Used to draw keypoints on an image
def draw_keypoints_on_image(image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
    """
    :param image: A numpy array with shape [height, width, 3]
    :param keypoints: A numpy array with shape [num_keypoints, 2]
    :param color: Color to draw the keypoints with
    :param radius: Keypoint radius
    :param use_normalized_coordinates: If True, treat keypoint values as relative to the image.
        Otherwise treat them as absolute
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]

    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])

    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius), (keypoint_x + radius, keypoint_y + radius)],
                     outline=color, fill=color)


# Method: Used to draw mask on an image (numpy array)
def draw_mask_on_image_array(image, mask, color='red', alpha=0.7):
    """
    :param image: uint8 numpy array with shape (img_height, img_height, 3)
    :param mask: uint8 numpy array of shape (img_height, img_height) with values between either 0 or 1
    :param color: Color to draw the keypoints with
    :param alpha: Transparency value between 0 and 1
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')

    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')

    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')

    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


# Method: Used to overlay labeled boxes on an image with formatted scores and label names
def visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index, instance_masks=None,
                                              keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=10,
                                              min_score_thresh=0.5, line_thickness=4):
    """
    :param image: A numpy array with shape (img_height, img_width, 3)
    :param boxes: A numpy array of shape [N, 4]
    :param classes: A numpy array of shape [N]. (Note: Class indices are 1-based, and match the keys in the label map)
    :param scores: A numpy array of shape [N] or None.  (Note: If scores=None, then this function assumes that the
        boxes to be plotted are groundtruth boxes and plot all boxes as black with no classes or scores)
    :param category_index: A dict containing category index `id` and category name `name`) keyed by category indices
    :param instance_masks: A numpy array of shape [N, image_height, image_width]
    :param keypoints: A numpy array of shape [N, num_keypoints, 2]
    :param use_normalized_coordinates: Whether boxes is to be interpreted as normalized coordinates or not
    :param max_boxes_to_draw: Maximum number of boxes to visualize.  If None, draw all boxes
    :param min_score_thresh: Minimum score threshold for a box to be visualized
    :param line_thickness: Integer controlling line width of the boxes
    """
    # Create a display string (and color) for every box location, group any boxes that correspond to the same location
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())

            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]

            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])

            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if classes[i] in category_index.keys():
                    class_name = category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'

                display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = standard_colors[classes[i] % len(standard_colors)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        box_center = (np.average(box[1::2]), np.average(box[0::2]))

        if instance_masks is not None:
            draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color)

        draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax, box_center, color=color, thickness=line_thickness,
                                         display_str_list=box_to_display_str_map[box],
                                         use_normalized_coordinates=use_normalized_coordinates)

        if keypoints is not None:
            draw_keypoints_on_image_array(image, box_to_keypoints_map[box], color=color, radius=line_thickness // 2,
                                          use_normalized_coordinates=use_normalized_coordinates)

    return image
