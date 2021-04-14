import pathlib
import numpy as np
import tensorflow as tf
from PIL import Image
from res_object_detection.utils import ops as utils_ops
from res_object_detection.utils import label_map_util
from six import viewkeys

# set image dimension constants
# this dependence is part of why new images
# don't work with our object detection network
im_height = 480
im_width = 640


def unnormalize(box, legend=False):
    """A function to add safe zones and create bounding boxes for each original image

    """

    if legend:
        safe_zone = 0
    else:
        safe_zone = 8
    ymin, xmin, ymax, xmax = box
    return ((xmin * im_width)-safe_zone, (ymin * im_height)-safe_zone, (xmax * im_width)+safe_zone, (ymax * im_height)+safe_zone)


def load_model():
    """A function to load our already trained object detection
        model from its directory and returns the model

    """
    model = tf.saved_model.load('fine_tuned_model/saved_model')
    model = model.signatures['serving_default']
    return model

# path to our label map
PATH_TO_LABELS = 'label_map.pbtxt'
# create categories from label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Path to images directory (where jpg conversions to be used for obj detection are stored)
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg"))) #note png / jpg difference
TEST_IMAGE_PATHS

# Load our detection model
detection_model = load_model()

detection_model.output_dtypes
detection_model.output_shapes


def run_inference_for_single_image(model, image):
    """A function to run the object detection on a given image

    Parameters:
    -----------
    model: a loaded tensorflow model
    image: an image array

    Returns:
    --------
    output_dict: dict, a dictionary containing the detection information from the image
    """

    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict


def record_boxes(image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None, #remove these unused arguments
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.2,
    agnostic_mode=False,
    skip_scores=False,
    skip_labels=False,
    ):
  """
  group boxes together that correspond ot the same location

  Input: image array, bounding boxes, object detection classes, scores
  Output: dictionary
  """

  box_dic = {}
  
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_dic):
      break
    if scores is None or scores[i] > min_score_thresh:
        box = tuple(boxes[i].tolist())
        display_class = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in viewkeys(category_index): #six.viewkeys
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_class = str(class_name)
        if not skip_scores:
          display_score = round(100*scores[i])

        if display_class not in box_dic:
          box_dic[display_class] = set()
        if display_class == 'legend':
            box_dic[display_class].add((unnormalize(box, legend=True), display_score))
        else:
            box_dic[display_class].add((unnormalize(box), display_score))

  box_dic['image_height'] = image.shape[0]
  box_dic['image_width'] = image.shape[1]
  
  return box_dic
      

def show_inference(model, image_path):
    """A function to properly call "run_inference_for_single_image"
        and "record_boxes" and return the resulting dictionary
    """

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    return record_boxes(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True)



def assign_labels(d):
    """A function to assign each object/class to its graph element
        Specifically distinguishing between the objects in the
        'text' class and sorting them into xlabel, ylabel, and title

    Call this function on the result of "show_inference()"

    Input:
    -----
    d: dict, a dictionary containing all of the results from the object detection

    Output:
    -------
    labels: dict, a dictionary with each bounding box as keys and the
        corresponding graph element as values
    """
    
    # initialize object detection classes to none
    labels = {}
    x_axis_label = None
    y_axis_label = None
    title_label = None
    legend = None

    # find x coordinate of the middle of the image
    img_mid_x = d['image_width']/2
    
    # assign each bounding box to a graph element
    if 'text' in d:
        max_x_displacement = 0
        min_y = d['image_height'] # because y is inverted
        for elem in d['text']:
            box,score = elem # here score is not actually used, but is left in for future applications
            cand = max(abs(box[0]-img_mid_x), abs(box[2]-img_mid_x))
            if cand > max_x_displacement:
                max_x_displacement = cand
                y_axis_label = box
            if box[3] < min_y:
                min_y = box[3]
                title_label = box
        for i,elem in enumerate(d['text']):
            box, score = elem
            if box != y_axis_label and box != title_label:
                x_axis_label = box
    labels[x_axis_label] = 'x axis'
    labels[y_axis_label] = 'y axis'
    labels[title_label] = 'title'
    if 'legend' in d:
        for elem in d['legend']:
            box,score = elem # here score is not actually used, but is left in for future applications
            legend = box
    labels[legend] = 'legend' # takes the last legend if multiple, but shouldn't be an issue since there should only ever be one
    return labels
