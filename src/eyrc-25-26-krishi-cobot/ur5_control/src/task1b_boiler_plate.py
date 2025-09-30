#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
*        		===============================================
*           		    Krishi coBot (KC) Theme (eYRC 2025-26)
*        		===============================================
*
*  This script should be used to implement Task 1B of Krishi coBot (KC) Theme (eYRC 2025-26).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boiler_plate.py
# Functions:
#			        [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#			        Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/aligned_depth_to_color/image_raw, /etc... ]



import sys
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import cv2

# runtime parameters
SHOW_IMAGE = True
DISABLE_MULTITHREADING = False

class FruitsTF(Node):
    """
    ROS2 Boilerplate for fruit detection and TF publishing.
    Students should implement detection logic inside the TODO sections.
    """

    def __init__(self):
        super().__init__('fruits_tf')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # callback group handling
        if DISABLE_MULTITHREADING:
            self.cb_group = MutuallyExclusiveCallbackGroup()
        else:
            self.cb_group = ReentrantCallbackGroup()

        # Subscriptions
        self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10, callback_group=self.cb_group)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depthimagecb, 10, callback_group=self.cb_group)

        # Timer for periodic processing
        self.create_timer(0.2, self.process_image, callback_group=self.cb_group)

        if SHOW_IMAGE:
            cv2.namedWindow('fruits_tf_view', cv2.WINDOW_NORMAL)

        self.get_logger().info("FruitsTF boilerplate node started.")

    # ---------------- Callbacks ----------------
    def depthimagecb(self, data):
        '''
        Description:    Callback function for aligned depth camera topic. 
                        Use this function to receive image depth data and convert to CV2 image.

        Args:
            data (Image): Input depth image frame received from aligned depth camera topic

        Returns:
            None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 
        #   -> Use `data` variable to convert ROS Image message to CV2 Image type
        #   -> HINT: You may use CvBridge to do the same
        #   -> Store the converted image into `self.depth_image`

        ############################################


    def colorimagecb(self, data):
        '''
        Description:    Callback function for colour camera raw topic.
                        Use this function to receive raw image data and convert to CV2 image.

        Args:
            data (Image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            None
        '''

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :
        #   -> Use `data` variable to convert ROS Image message to CV2 Image type
        #   -> HINT: You may use CvBridge to do the same
        #   -> Store the converted image into `self.cv_image`
        #   -> Check if you need any rotation or flipping of the image 
        #      (as input data may be oriented differently than expected).
        #      You may use cv2 functions such as `cv2.flip` or `cv2.rotate`.

        ############################################


    def bad_fruit_detection(self, rgb_image):
        '''
        Description:    Function to detect bad fruits in the image frame.
                        Use this function to detect bad fruits and return their center coordinates, distance from camera, angle, width and ids list.

        Args:
            rgb_image (cv2 image): Input coloured raw image frame received from image_raw camera topic

        Returns:
            list: A list of detected bad fruit information, where each entry is a dictionary containing:
                - 'center': (x, y) coordinates of the fruit center
                - 'distance': distance from the camera in meters
                - 'angle': angle of the fruit in degrees
                - 'width': width of the fruit in pixels
                - 'id': unique identifier for the fruit
        '''
        ############ ADD YOUR CODE HERE ############
        # INSTRUCTIONS & HELP :
        #   ->  Implement bad fruit detection logic using image processing techniques
        #   ->  You may use techniques such as color filtering, contour detection, etc.
        #   ->  For each detected bad fruit, create a dictionary with its information and append
        #       to the bad_fruits list
        #   ->  Return the bad_fruits list at the end of the function
        # Step 1: Convert RGB image to HSV color space
        #   - Use cv2.cvtColor to convert the input image to HSV for better color segmentation

        # Step 2: Define lower and upper HSV bounds for "bad fruit" color
        #   - Choose HSV ranges that correspond to the color of bad fruits (e.g., brown/black spots)

        # Step 3: Create a binary mask using cv2.inRange
        #   - This mask highlights pixels within the specified HSV range

        # Step 4: Find contours in the mask
        #   - Use cv2.findContours to detect continuous regions (potential bad fruits)

        # Step 5: Loop through each contour
        #   - Filter out small contours by area threshold to remove noise
        #   - For each valid contour:
        #       a. Compute bounding rectangle (cv2.boundingRect)
        #       b. Calculate center coordinates (cX, cY)
        #       c. (Optional) Calculate distance and angle if depth data is available
        #       d. Store fruit info (center, distance, angle, width, id) in a dictionary
        #       e. Append dictionary to bad_fruits list

        # Step 6: Return the bad_fruits list
        bad_fruits = []

        # TODO: Implement bad fruit detection logic here
        # You may use image processing techniques such as color filtering, contour detection, etc.
        # For each detected bad fruit, append its information to the bad_fruits list

        return bad_fruits


    def process_image(self):
        '''
        Description:    Timer-driven loop for periodic image processing.

        Returns:
            None
        '''
        ############ Function VARIABLES ############

        # These are the variables defined from camera info topic such as image pixel size, focalX, focalY, etc.
        # Make sure you verify these variable values once. As it may affect your result.
        # You can find more on these variables here -> http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 642.724365234375
        centerCamY = 361.9780578613281
        focalX = 915.3003540039062
        focalY = 914.0320434570312
            

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP : 

        #   ->  Get fruit center, distance from rgb, angle, width and ids list from 'detect_fruit_center' defined above

        #   ->  Loop over detected box ids received to calculate position and orientation transform to publish TF 

        #   
        #   ->  Use center_fruit_list to get realsense depth and log them down.

        #   ->  Use this formula to rectify x, y, z based on focal length, center value and size of image
        #       x = distance_from_rgb * (sizeCamX - cX - centerCamX) / focalX
        #       y = distance_from_rgb * (sizeCamY - cY - centerCamY) / focalY
        #       z = distance_from_rgb
        #       where, 
        #               cX, and cY from 'center_fruit_list'
        #               distance_from_rgb is depth of object calculated in previous step
        #               sizeCamX, sizeCamY, centerCamX, centerCamY, focalX and focalY are defined above

        #   ->  Now, mark the center points on image frame using cX and cY variables with help of 'cv2.circle' function 

        #   ->  Here, till now you receive coordinates from camera_link to fruit center position. 
        #       So, publish this transform w.r.t. camera_link using Geometry Message - TransformStamped 
        #       so that we will collect its position w.r.t base_link in next step.
        #       Use the following frame_id-
        #           frame_id = 'camera_link'
        #           child_frame_id = 'cam_<fruit_id>'          Ex: cam_20, where 20 is fruit ID

        #   ->  Then finally lookup transform between base_link and obj frame to publish the TF
        #       You may use 'lookup_transform' function to pose of obj frame w.r.t base_link 

        #   ->  And now publish TF between object frame and base_link
        #       Use the following frame_id-
        #           frame_id = 'base_link'
        #           child_frame_id = f'{teamid}_bad_fruit_{fruit_id}'    Ex: 5_bad_fruit_1, where 5 is team ID and 1 is fruit ID

        #   ->  At last show cv2 image window having detected markers drawn and center points located using 'cv2.imshow' function.
        #       Refer MD book on portal for sample image -> https://portal.e-yantra.org/


def main(args=None):
    rclpy.init(args=args)
    node = FruitsTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down FruitsTF")
        node.destroy_node()
        rclpy.shutdown()
        if SHOW_IMAGE:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
