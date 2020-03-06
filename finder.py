#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
## below added from RoboFarm for ObjectDetection/Image Classification
import os
import random
import time

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
# below imports are from RoboFarm 
# RoboFarm github (does not have data files) https://github.com/john-reilly/RoboFarm
# RoboFarm shared Google drive with data https://drive.google.com/drive/folders/1x4hxiDkYLmqQzikF6OZ6oFNLfvZROom2
#
#from google.colab.patches import cv2_imshow # You might not need this
from moviepy.editor import VideoFileClip
#from IPython.display import HTML 
import cv2

# Find more models here
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# for Ref How to Retrain here, too much to do today....
# https://www.tensorflow.org/hub/tutorials/image_retraining
# the definitive tutorial from Tensorflow
# https://github.com/tiangolo/tensorflow-models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# the COCO categories are here
# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

#base_dir = '/content/gdrive/My Drive/RoboFarm/'
# It's not best practice to put everything into /scripts folder but I do so here for ease of presenation in a workshop
base_dir = '/home/pycon/catkin_ws/src/beginner_tutorials/scripts/' # Leaving Base dir as current dir for simplicity
class ObjectDetector:
    
    def __init__(self,location_path = base_dir, graph_filename = 'frozen_inference_graph.pb' ):       
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef() # tensorflow 1.x way
            #od_graph_def = tf.compat.v1.GraphDef() 
            #from https://stackoverflow.com/questions/57614436/od-graph-def-tf-graphdef-attributeerror-module-tensorflow-has-no-attribut
            #tf.compat.v1.GraphDef()   # -> instead of tf.GraphDef()
            #tf.compat.v2.io.gfile.GFile()   # -> instead of tf.gfile.GFile()

            try:
                with tf.gfile.GFile( location_path + graph_filename, 'rb') as fid: #tensorflow 1.x way
                #with tf.compat.v2.io.gfile.GFile(location_path + graph_filename, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    self.session = tf.Session(graph=self.detection_graph) #old Tensorflow way
                    #self.session = tf.compat.v1.Session(graph=self.detection_graph) 
                    
            except Exception as e:
                print(e)
                exit()


    def run_inference_for_single_image(self, image, show_stats = True ):

        # Get handles to input and output tensors
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0') #indentation?

        # Run inference
        start = time.time()
        output_dict = self.session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        elapsed = time.time() - start
        #print('inference took:', elapsed, ' seconds') #optional, try with a single frame
        
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        if show_stats == True:
          print('inference took:', elapsed, ' seconds') 
          print("num_detections:" , output_dict['num_detections'])
          print("Detection Classes: " , output_dict['detection_classes'])
          print("Detection Scores: " , output_dict['detection_scores'])

        return output_dict

    
    def find_and_box(self, image, output_dict, draw_boxes = False, colour = 'Red'):
        classes = output_dict['detection_classes']
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']

        height, width, _ = image.shape

        ymin, xmin, ymax, xmax = tuple(boxes[0].tolist())
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        # <------ TODO how would you search for specific categories here?
        # <-------TODO How would you filter very small objects here?      
        detected_class = classes[0] 
            
        #draw rectangle around object if draw_boxes == True and choice of colour           
        if draw_boxes:
          if colour == 'Red' :
            #OpenCV is BGR Blue Green Red colour scheme
            cv2.rectangle(image, (xmin,ymin) , (xmax, ymax) , (0,0,255) ,4) 
          elif colour == 'Blue' :
            cv2.rectangle(image, (xmin,ymin) , (xmax, ymax) , (255,0,0) ,4)
          elif colour == 'Green' :
            cv2.rectangle(image, (xmin,ymin) , (xmax, ymax) , (0,255,0) ,4)
        
        return image, detected_class
    
def get_random_image():

    number_of_images = 4 # there are 4 images in the src folder to choose from
    # <------- TODO TRY adding your own images to see what happens check outputs for new images
    random_value = random.randint(0, number_of_images -1)
    print(random_value)

    if random_value == 0:
      file_path = base_dir + 'apple.jpg'

    elif random_value == 1:
      file_path = base_dir + 'banana.jpg'

    elif random_value == 2:
      file_path = base_dir + 'broccoli.jpg'

    elif random_value == 3:
      file_path = base_dir + 'orange.jpg'

    random_image =  cv2.imread(file_path,-1)
    #random_image = cv2.imread(base_dir + 'traffic_light.jpg') # This was to manualy select the traffic light file
    #random_image = cv2.imread(base_dir + 'Beans_2.jpg') # This was to manualy select the beans file
    #print"filepath:" + file_path # This line is to check the file path
    
    return (random_image)

def display_image(image, window_label, delay_in_miliseconds):
  
    cv2.imshow( window_label, cv2.resize(image, None, fx = 0.50 , fy = 0.50, interpolation = cv2.INTER_CUBIC) )
    
    cv2.waitKey(delay_in_miliseconds)

    return

def finder():

    # below from ROS.org talker.py
    pub = rospy.Publisher('detected_objects', String, queue_size=10)
    rospy.init_node('finder', anonymous=True)
    rate = rospy.Rate(.05)#0) # 1 is 1 Hz BE CAREFUL HERE ! rate must allow display windows etc to open and close
    # make a new classifier object, 'frozen_inference_graph.pb' is  a pretrained classifer using COCO dataset
    my_detector = ObjectDetector(base_dir , 'frozen_inference_graph.pb')
    
    # This is the main loop
    while not rospy.is_shutdown():

        random_image = get_random_image()
        display_image(random_image, 'Before', 1000) # 'before' is window label, 1000 is delay in milisecond
        
        # with OPEN CV and Tensorflow you need to flip the colour scheme
        # <----- TODO Try not changing colours to see how it effects result
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        
        # RUN object decttor this is called inferance
        output_dict = my_detector.run_inference_for_single_image(random_image, show_stats = True)# stats are the discovered classes and their probabilites
        
        
        final_image, detected_class = my_detector.find_and_box( random_image, output_dict, draw_boxes = True, colour = 'Red')
        #flip colours again
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        display_image(final_image, 'After', 1000) 
        
        message =  str(detected_class) # ROS needs you to explictly change the type
        rospy.loginfo(message) 
        pub.publish(message) 
        rate.sleep()

if __name__ == '__main__':
    try:
        finder()
    except rospy.ROSInterruptException:
        pass
