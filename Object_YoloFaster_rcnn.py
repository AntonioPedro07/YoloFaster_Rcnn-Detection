#!/usr/bin/python3

import argparse
import queue
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import roypy
import torch
import torchvision

from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights

from roypy_sample_utils import CameraOpener, add_camera_opener_options, select_use_case
from roypy_platform_utils import PlatformHelper
from PIL import Image

classes = {
    0: u"__background__",
    1: u"person",
    2: u"bicycle",
    3: u"car",
    4: u"motorcycle",
    5: u"airplane",
    6: u"bus",
    7: u"train",
    8: u"truck",
    9: u"boat",
    10: u"traffic light",
    11: u"fire hydrant",
    12: u"stop sign",
    13: u"parking meter",
    14: u"bench",
    15: u"bird",
    16: u"cat",
    17: u"dog",
    18: u"horse",
    19: u"sheep",
    20: u"cow",
    21: u"elephant",
    22: u"bear",
    23: u"zebra",
    24: u"giraffe",
    25: u"backpack",
    26: u"umbrella",
    27: u"handbag",
    28: u"tie",
    29: u"suitcase",
    30: u"frisbee",
    31: u"skis",
    32: u"snowboard",
    33: u"sports ball",
    34: u"kite",
    35: u"baseball bat",
    36: u"baseball glove",
    37: u"skateboard",
    38: u"surfboard",
    39: u"tennis racket",
    40: u"bottle",
    41: u"wine glass",
    42: u"cup",
    43: u"fork",
    44: u"knife",
    45: u"spoon",
    46: u"bowl",
    47: u"banana",
    48: u"apple",
    49: u"sandwich",
    50: u"orange",
    51: u"broccoli",
    52: u"carrot",
    53: u"hot dog",
    54: u"pizza",
    55: u"donut",
    56: u"cake",
    57: u"chair",
    58: u"couch",
    59: u"potted plant",
    60: u"bed",
    61: u"dining table",
    62: u"toilet",
    63: u"tv",
    64: u"laptop",
    65: u"mouse",
    66: u"remote",
    67: u"keyboard",
    68: u"cell phone",
    69: u"microwave",
    70: u"oven",
    71: u"toaster",
    72: u"sink",
    73: u"refrigerator",
    74: u"book",
    75: u"clock",
    76: u"vase",
    77: u"scissors",
    78: u"teddy bear",
    79: u"hair drier",
    80: u"toothbrush",
    81: u"hand",
    82: u"pen",
    83: u"key",
    84: u"tape",
    85: u"paper",
    86: u"box",
    87: u"notebook",
    88: u"monitor",
    89: u"wallet",
    90: u"table",
    91: u"smartwatch",
}

model = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# insert the path to your Royale installation here:
# note that you need to use \\ or / instead of \ on Windows
ROYALE_DIR = "C:/Program Files/royale/5.4.0.2112/python"
sys.path.append(ROYALE_DIR)

# load the COCO class labels our YOLO model was trained on
CLASSES = None
with open("coco.names", 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

def get_output_layers(net):

    # get the names of all layers in the network
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# function to get the distance to the object
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, distance):

    x = int(round(x))
    y = int(round(y))
    x_plus_w = int(round(x_plus_w))
    y_plus_h = int(round(y_plus_h))

    # draw a bounding box rectangle and label on the image
    label = "{} : {:.2f} - Distance: {:.2f} m".format(CLASSES[class_id], confidence, distance)
    color = COLORS[class_id]
    # print(x, y, x_plus_w, y_plus_h) # DEBUG
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
def emitAlert(distance, object_name):

    """
    Emits an alert based on the distance of the detected object.
    :param distance: The distance to the object in meters.
    :param object_name: The name/class of the detected object.
    """
    INFO_DISTANCE = 1.0  # Distance to inform about the object's presence
    DANGER_DISTANCE = 0.5  # Distance considered as dangerous
    TOLERANCE = 0.05 # Tolerance for considering the distance as 1 meter

    if distance <= DANGER_DISTANCE: 
        print(f"Immediate danger! Object '{object_name}' too close: {distance:.2f}m")
        return True
    elif INFO_DISTANCE - TOLERANCE <= distance <= INFO_DISTANCE + TOLERANCE:
        print(f"Object detected: '{object_name}', at {distance:.2f}m distance.")
        return True
    return False

def processDepthImage(depth_image):

    # Normalize the depth image to be of type uint8
    depthImgNormalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized depth image to uint8
    depthImg_uint8 = np.uint8(depthImgNormalized)

    # Convert the uint8 depth image to BGR
    depthImgGray = cv2.cvtColor(depthImg_uint8, cv2.COLOR_GRAY2BGR)

    return depthImgGray

def nmsObjectsRcnn(boxes, scores, iou_threshold):
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    x1 = boxes [:, 0]
    y1 = boxes [:, 1]
    x2 = boxes [:, 2]
    y2 = boxes [:, 3]

    # compute the area of the bounding boxes and sort the bounding
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    # keep looping while some indexes still remain in the indexes
    # list
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute the coordinates of the intersection rectangle
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
    # compute the width and height of the bounding box
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
    # keep only the indexes of the bounding boxes that
        # overlap significantly with the bounding box
        # corresponding to the first bounding box
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# Calculate the average distance of the object
def calculateAverageDistance(depth_map, x1, y1, x2, y2):
    # Get the depth values of the region of interest
    depth_region = depth_map[y1:y2, x1:x2]

    # Get the average depth value of the region of interest
    positive_depth = depth_region[depth_region > 0]

    # Calculate the average distance of the object
    if positive_depth.size > 0:
        averageDistance = np.mean(positive_depth) # Average distance in meters
    else:
        averageDistance = float('inf') # If there is no depth value, return inf

    return averageDistance

# Detect objects using the RCNN model
def detectObjectsRcnn(frame, model, device, depth_map):
    # Transform the image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(frame).to(device).unsqueeze(0)
    alert_issue = False

    with torch.no_grad():
        output = model(image)
    
    # Get all the predicited class names
    thres = 0.65
    scores = output[0]['scores'].detach().cpu().numpy()
    hight_scores_idxs = np.where(scores > thres)[0]
    hight_scores_boxes = output[0]['boxes'].detach().cpu().numpy()[hight_scores_idxs]
    hight_scores_labels = output[0]['labels'].detach().cpu().numpy()[hight_scores_idxs]

    # Apply non-max suppression to suppress weak, overlapping bounding boxes
    after_nms = nmsObjectsRcnn(hight_scores_boxes, scores[hight_scores_idxs], iou_threshold = 0.4)

    # Get the filtered boxes, labels and scores
    filtered_boxes = hight_scores_boxes[after_nms]
    filtered_labels = hight_scores_labels[after_nms]
    filtered_scores = scores[hight_scores_idxs][after_nms]

    # Draw bounding boxes and labels of detections
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        # Transform the coordinates to the original image
        x1, y1, x2, y2 = map(int, box)
        averageDistance = calculateAverageDistance(depth_map, x1, y1, x2, y2)

        #if label in classInterest: # Get the class name using Id
        try:
            label_name = classes[label] # Get the class name
            confidence = scores[np.where(hight_scores_labels == label)[0][0]]
        except KeyError:
                print(f"Label {label} not found.")
                continue

        if emitAlert(averageDistance, label_name):
            alert_issue = True
        if not alert_issue:
            # Get the distance from the camera
            print(f"Object detected: {label_name} - {confidence:.2f} - Distance: {averageDistance:.2f} meters")

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print(f"Box draw for {label_name} : {(x1, y1, x2, y2)}") # Debug: Confirm if boxes are being drawn
    
    return frame

def detectObjects(img, depth_map):
    #  Get the image dimensions 
    Width = img.shape[1]
    Height = img.shape[0]
    classInterest = [0, 24, 26, 39, 64, 66, 67, 74]
    scale = 1/255
    
    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), False, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net)) # Forward pass throught the network

    # Initialization for late use
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.65 #  Confidence threshold
    nms_threshold = 0.4 #  Non-maximum suppression threshold

    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:] # Get scores of all scores 
            class_id = np.argmax(scores) # Class with the  highest score
            confidence = scores[class_id] # Confidence of the prediction
            if confidence > 0.5: # Filter out weak detections

                # Calculate bounding box coordinates
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = detection[2] * Width
                h = detection[3] * Height
                x = int (center_x - w / 2)
                y = int (center_y - h / 2)

                # Append to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply Non-max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    if isinstance(indices, np.ndarray):
        indices = indices.flatten().tolist()
    elif isinstance(indices, tuple):
        indices = []

    if len(indices) == 0:
        
        # Convert to RGB, since opencv uses BGR
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame = np.array(pil_image) # frame is now RGB

        # Process the model
        detectObjectsRcnn(frame, model, device, depth_map)

        # Convert back to BGR, since opencv uses BGR and PIL uses RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    

    alert_issued = False

    for i in indices:

        class_id = class_ids[i]
        # Check if the object is in the list of objects of interest
        if class_id in classInterest:
            # Get the bounding box coordinates
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

        # Calculate the average distance of the object
        depth_region = depth_map[int(y): int(y + h), int(x): int(x + w)]
        positiveDepthValues = depth_region[depth_region > 0]

        # If there are no positive depth values, then the object is too far away
        if positiveDepthValues.size > 0:
            average_distance = np.mean(positiveDepthValues) # Ignore zero values, if applicable
        else:
            average_distance = float('inf')
        distance_meters = average_distance # Adjust as necessary for the correct unit

        #Draw predictions on the image
        draw_prediction(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), distance_meters)

        # Get the name of the object
        object_name = CLASSES[class_ids[i]]

        if emitAlert(distance_meters, object_name):  # Calls the alert function with the distance and detected object
            alert_issued = True

        if not alert_issued:
            #Print the detection in the console
            print(f"Object detected: {CLASSES[class_ids[i]]} at Distance: {distance_meters:.2f} meters!")

    return img

# OPENCV SAMPLE + INTEGRATED OBJECT DETECTION WITH YOLO
class MyListener(roypy.IDepthDataListener):
    
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = True
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)

    def paint (self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """

        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]
        gray = data[:, :, 3]
        confidence = data[:, :, 4]

        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)
        depthImgRaw = depth
        depthImgProcess = processDepthImage(depthImgRaw)  

        # iterate over matrix, set zImage values to z values of data
        # also set grayImage adjusted gray values
        xVal = 0
        yVal = 0
        for x in zImage:        
            for y in x:
                if confidence[xVal][yVal]> 0:
                  grayImage[xVal,yVal] = self.adjustGrayValue(gray[xVal][yVal])
                yVal=yVal+1
            yVal = 0
            xVal = xVal+1

        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        if self.undistortImage: 
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # convert the image to rgb first, because YOLO needs 3 channels, and then detect the objects
        yoloResultImageGray = detectObjects(cv2.cvtColor(grayImage8, cv2.COLOR_GRAY2RGB), depth)
        yoloResultDepthImage = detectObjects(depthImgProcess, depthImgRaw)
        
        # finally show the images
        cv2.imshow("Gray Image", yoloResultImageGray)
        cv2.imshow("Depth Image", yoloResultDepthImage)

        self.lock.release()
        self.done = True

    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the gray values from the camera to 0..255
    def adjustGrayValue(self,grayValue):
        clampedVal = min(400,grayValue) # try different values, to find the one that fits your environment best
        newGrayValue = clampedVal / 400 * 255
        return newGrayValue

def main ():
    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
    opener = CameraOpener (options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)

    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue (q, l)

    cam.stopCapture()
    print("Done")

def process_event_queue (q, painter):

    while True:
        try:

            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item) 
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            """print(f"Current  key pressed: {currentKey}")""" #for debuging purposes
            if currentKey == ord('d'):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27: 
                break

if (__name__ == "__main__"):
    main()
