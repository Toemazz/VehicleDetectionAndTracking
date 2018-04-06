import logging
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2


# Method: Used to checks if a label map is valid
def validate_label_map(label_map):
    """
    :param label_map: StringIntLabelMap to validate
    :return: ValueError: if label map is invalid
    """
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map ids should be >= 1.')


# Method: Creates a dictionary of COCO compatible categories keyed by category id
def create_category_index(categories):
    """
    :param categories: A list of dicts, each of which has the following keys: 'id', 'name'
    :return: A dict containing the same entries as categories, but keyed by the 'id' field of each category
    """
    category_index = {}

    for cat in categories:
        category_index[cat['id']] = cat

    return category_index


# Method: Loads label map proto and returns categories list compatible with eval
def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    """
    :param label_map: A StringIntLabelMapProto or None.  If None, a default categories list is created with
        max_num_classes categories.
    :param max_num_classes: Maximum number of (consecutive) label indices to include
    :param use_display_name: Choose whether to load 'display_name' field as category name
    :return: A list of dictionaries representing all possible categories
    """
    categories, list_of_ids_already_added = [], []

    if not label_map:
        label_id_offset = 1

        for class_id in range(max_num_classes):
            categories.append({'id': class_id + label_id_offset,
                               'name': 'category_{}'.format(class_id + label_id_offset)})
        return categories

    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested label range.', item.id)
            continue

        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name

        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})

    return categories


# Method: Used to load label map proto
def load_label_map(path):
    """
    :param path: Path to StringIntLabelMap proto text file
    :return: A StringIntLabelMapProto
    """
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()

        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)

    validate_label_map(label_map)

    return label_map


# Method: Used to read a label map and returns a dictionary of label names to id
def get_label_map_dict(label_map_path, use_display_name=False):
    """
    :param label_map_path: Path to label_map
    :param use_display_name: Whether to use the label map items' display names as keys
    :return: A dictionary mapping label names to id
    """
    label_map = load_label_map(label_map_path)
    label_map_dict = {}

    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id

    return label_map_dict


# Method: Used to read a label map and returns a category index
def create_category_index_from_label_map(label_map_path):
    """
    :param label_map_path: Path to `StringIntLabelMap` proto text file
    :return: A category index, which is a dictionary that maps integer ids to dicts containing categories
    """
    label_map = load_label_map(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    categories = convert_label_map_to_categories(label_map, max_num_classes)

    return create_category_index(categories)
