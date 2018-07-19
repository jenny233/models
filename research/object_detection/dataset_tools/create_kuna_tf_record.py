import os
import json
import csv
from PIL import Image
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('pdm_file', None, 'Path to the .pdm file')
flags.DEFINE_string('images_dir', None, 'Directory containing all images')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord')
FLAGS = flags.FLAGS

def size_to_coords(x, y, w, h):
  '''
  x, y are the coordinates of the center of the box
  w, h are width and height of the box
  Returns a tuple of (xmin, xmax, ymin, ymax)
  '''
  xmin = x - w / 2.0
  xmax = x + w / 2.0
  ymin = y - h / 2.0
  ymax = y + h / 2.0
  return (xmin, xmax, ymin, ymax)

def create_tf_example(image):

  category_to_class = {
      # person
      "PERSON": 1,
      # vehicle
      "STATIONARY_CAR": 3,
      "MOVING_CAR": 3,
  }
  # Filename of the image. Needs to be bytestring
  filename = os.path.join(FLAGS.images_dir, image['path']).encode('latin-1') 
  with Image.open(filename) as image_file:
    width, height = image_file.size # Image width and height
  with tf.gfile.Open(filename) as image_file:
    encoded_image_data = image_file.read() # Encoded image bytes

  image_format = b'jpeg'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)

  # Get the list of boxes from .pdm file (as a dictionary)
  categoryBoxes = image["categoryBoxes"]
  # For every box in this image
  for categoryBoxId in categoryBoxes:
      box = categoryBoxes[categoryBoxId]
      x = box["x"]
      y = box["y"]
      w = box["width"]
      h = box["height"]
      category = box["category"]
      (xmin, xmax, ymin, ymax) = size_to_coords(x, y, w, h)
      xmins.append(xmin)
      xmaxs.append(xmax)
      ymins.append(ymin)
      ymaxs.append(ymax)
      # class text and class id should be determined by label map
      # default use mscoco_label_map.pbtxt
      if category_to_class[category] == 1:
        classes_text.append('person')
        classes.append(1)
      elif category_to_class[category] == 3:
        classes_text.append('car')
        classes.append(3)
      else:
        classes_text.append('unknown')
        classes.append(0)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):

  required_flags = [
      'pdm_file', 'images_dir', 'output_path',
  ]
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  FLAGS.images_dir = os.path.abspath(FLAGS.images_dir)
  FLAGS.pdm_file = os.path.abspath(FLAGS.pdm_file)


  # TODO(user): Write code to read in your dataset to examples variable
  with open(FLAGS.pdm_file, "r") as f:
    data = json.load(f)
    images = data["images"]
    for image_key in images:
      image = images[image_key]
      # If image isFullyCategorized then it is added to the dataset
      if image["isFullyCategorized"]:
        tf_example = create_tf_example(image)
        writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
