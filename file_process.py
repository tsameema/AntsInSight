import albumentations as A
import cv2, os, time
from result import PREDICT_MODEL

class FILE_PROCESS():
  def __init__(self, file, test_label_path, save_folder):
    """
      Initialize the FileProcessor with file, test_label_path, and save_folder.

      Args:
          file (BytesIO): The input image file in BytesIO format.
          test_label_path (str): The path to test label data.
          save_folder (str): The folder where processed files will be saved.
    """
    self.file = file
    self.test_label_path = test_label_path
    self.save_folder=save_folder

  def validate_file(self):
    """
        Validate the input file and save it to the specified folder.

        Returns:
            str: The saved image file path.
            str: The original file name.
    """
    file_name = self.file.name
    if file_name.lower().endswith(".png"):
      name = str(time.time()).split(".")[0]
      imgfile = os.path.join(self.save_folder, f"{name}.png")
      with open(imgfile, "wb") as f:
          f.write(self.file.getvalue())
      return imgfile, file_name
    else:
      return None, None  #if file extension is not .png then return None
    
  def test_model(self, imagefile, filename):
      """
        Test a model on the provided image file.

        Args:
            image_file_path (str): The path to the image file.
            original_filename (str): The original file name.

        Returns:
            None
      """
      image, label_file = self.applying_albumentation(imagefile, filename, self.test_label_path, self.save_folder)

      results = PREDICT_MODEL.predict_image(image)
      PREDICT_MODEL.display_results(results, image, label_file)

  def applying_albumentation(self, image_file, filename, test_label_path, save_folder):
      
      """
        Apply Albumentations to an image and its associated label file.

        Args:
            image_file (str): Path to the input image file.
            filename (str): The original file name.
            test_label_path (str): Path to the test label data.
            save_folder (str): Folder where processed files will be saved.

        Returns:
            cv2.imread : The transformed image.
            str: Path to the saved transformed label file.
      """
      A_HEIGHT, A_WIDTH = 512, 640
      kp_labels = ['abdomen', 'head']
      kp_sides = ['top', 'bottom']
      bbox_labels = ['ant']

      #reading image and label file
      img_read = cv2.imread(image_file)
      HEIGHT, WIDHT = img_read.shape[0], img_read.shape[1]
      kps_read = self.read_label_file(os.path.join(test_label_path, f"{filename.split('.')[0]}.txt"))

      #transformation pipeline
      transform = self.create_albumentation_pipeline(kp_labels, kp_sides, bbox_labels)

      #saving the transformed labels into txt file
      save_albument_label = os.path.join(save_folder, f"{filename.split('.')[0]}.txt")
      with open(save_albument_label, "w+") as f:
          for kp_read in kps_read:
              
              transformed = transform(image = img_read,
                                      keypoints = [kp_read[5:7], kp_read[7:]],
                                      bboxes = [self.converted_bbox([kp_read[1:5]], HEIGHT, WIDHT)],
                                      bbox_labels = bbox_labels,
                                      kp_labels = kp_labels,
                                      kp_sides = kp_sides
                                      )
              
              transform_kp = [((k[0]/A_WIDTH), (k[1]/A_HEIGHT)) for k in transformed['keypoints']]
              data = kp_read[0], transformed['bboxes'], transform_kp
              f.write(self.combine_cls_bbox_kp(data)+'\n')
      f.close()

      return transformed['image'], save_albument_label 

  def read_label_file(self, path):
    """
    Reads ground truth label file, convert the data from string to int, and remove all the
    unnecessary text
    Args:
        file (str): path of label file
    Returns:
        list: A list of ground truth in formated way

    """
    with open(path, 'r') as f:
          lines = f.read().strip().split('\n')
          keypoint = [list(map(int, line.split())) for line in lines]
    return keypoint

  def combine_cls_bbox_kp(self, data):
    """
    Combines class, bbox, and keypoint labels into a single line after applying Albumentations
    transformations.

    Args:
          data (tuple): A tuple containing class label, bounding box, and keypoint data.

    Returns:
          str: A string containing the combined labels in a single line.

    """
    class_data, bbox_data, keypoint_data = data
    all_labels = [str(class_data)]

    for bbox in bbox_data[0]:
      all_labels.append(str(bbox))

    for key_pt in keypoint_data:
      all_labels.extend(map(str, key_pt))

    return ' '.join(all_labels)

  def converted_bbox(self, bbox, img_height, img_width):
    """
    convert the bbox into required YOLO format
    given : [xmin, xmax, ymin, ymax]
    required : [xcenter, ycenter, width, height]

    Args:
      bbox (list) : A list containing bounding box coordinates [xmin, xmax, ymin, ymax]
      img_height (int) : Height of image
      img_width (int)  : Width of image

    Returns:
      tuple :  A tuple containing YOLO format bounding boxes coordinates [xcenter, ycenter, width, height]
    """
    x_min, y_min, x_max, y_max = bbox[0]
    x_center = (( x_min + x_max ) / 2 ) / img_width
    y_center = (( y_min + y_max ) / 2 ) / img_height
    width = ( x_max - x_min) / img_width
    height = ( y_max - y_min ) / img_height
    return (x_center), (y_center), (width), (height)
    

  def create_albumentation_pipeline(self, kp_labels, kp_sides, bbox_labels):
    """
    Create an albumentation pipeline as image is resized

    Args:
      kp_label (list) : A list contains labels for the keypoints either abdomen or head
      kp_sides (list) : A list takes sides of the keypoints either top or bottom
      bbox_labels (list) : A list contains class to which bbox belongs

    Returns:
      A.Compose: An Albumentations pipeline for image augmentation.

    """

    # Define image dimensions
    A_HEIGHT, A_WIDTH = 512, 640
    
    # Create the Albumentations pipeline
    transform =  A.Compose([
        A.Resize(A_HEIGHT, A_WIDTH)
        ],
        bbox_params=A.BboxParams(
            format='yolo', 
            label_fields = ['bbox_labels'],
        ),
        keypoint_params=A.KeypointParams(
            format='xy', 
            label_fields=['kp_labels', 'kp_sides'],
            remove_invisible=True,
        ))
    return transform
        


