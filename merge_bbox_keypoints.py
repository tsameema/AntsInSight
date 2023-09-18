import os, re

# Merge the bbox data and keypoints into one file and add the classes each bounding box has.
# Here, we have just one class "ant" so 0 is added at the start of each line.
# The format of the labels file is as follows:
#
# class [bbox] [keypoints]
# [class] [[xmin, ymin, xmax, ymax]] [[xa, ya, xh, yh]]
#
# xmin, ymin - coordinates of the top-left point of the bounding box
# xmax, ymax - coordinates of the lower-right point of the bounding box
# xa, ya - abdomen center coordinates
# xh, yh - head center coordinates.


box_path = 'archive/data/Train_data/box'
key_path = 'archive/data/Train_data/keypoints'

# Iterate through the files in the box_path directory
for box in os.listdir(box_path):
    # Check if the corresponding keypoint file exists
    keypoint_file = box.replace('bbox', 'keypoint')
    if keypoint_file in os.listdir(key_path):
        # Open the bbox and keypoint files
        with open(os.path.join(box_path, box), 'r') as bbox_file, open(os.path.join(key_path, keypoint_file), 'r') as keypoint_file:
            # Read the lines from both files
            bbox_lines = bbox_file.readlines()
            keypoint_lines = keypoint_file.readlines()
        
        # Merge the data and add the class label (0)
        merged_lines = ['0 ' + bbox_line.strip() + ' ' + keypoint_line for bbox_line, keypoint_line in zip(bbox_lines, keypoint_lines)]
        
        # Write the merged data back to the bbox file
        with open(os.path.join(box_path, box), 'w') as merged_bbox_file:
            merged_bbox_file.writelines(merged_lines)


def rename_files(path, find_word):
    """
        Image and labels are renamed

    ARGS:
        path (str) : Path of either image or labels
        find_word (str) : A text that need to be replaced

    RETURNS:
        None
    """
    for mode in ['train', 'val', 'test']:
        for f in os.listdir(os.path.join(path, mode)):
            if re.search(fr"{find_word}", f):
                new_name = re.sub(fr"{find_word}", "000", f)
                old_path = os.path.join(path, mode, f)
                new_path = os.path.join(path, mode, new_name)
                os.rename(old_path, new_path)


img_path = "archive/ant_bbox_keypoint/images"
rename_files(img_path, "image")

label_path = "archive/ant_bbox_keypoint/labels"
rename_files(label_path, "bbox_keypoint_")

