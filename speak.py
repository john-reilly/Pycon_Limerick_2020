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

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic
#from ROS
import rospy
from std_msgs.msg import String
#for speech 
import pyttsx3

# This is a list of the COCO dataset categories using the standard labels
COCO_categories = [
 'person' ,'bicycle','car','motorcycle','airplane',  
 'bus','train','truck','boat','traffic light',
 'fire hydrant', 'street sign','stop sign','parking meter','bench',
 'bird','cat','dog','horse','sheep', 
 'cow','elephant','bear','zebra','giraffe',
 'hat','backpack','umbrella','shoe','eye glasses', 
 'handbag' ,'tie','suitcase','frisbee','skis',
 'snowboard','sports','kite','baseball bat','baseball glove',
 'skateboard','surfboard','tennis racket','bottle','plate',
 'wine glass' ,'cup','fork','knife','spoon',
 'bowl','banana','apple','sandwich','orange',
 'broccoli', 'carrot','hot dog','pizza','donut', 
 'cake','chair','couch','potted plant','bed',
 'mirror','dining table','window','desk','toilet',
 'door','tv','laptop','mouse','remote', 
 'keyboard','cell phone','microwave','oven','toaster',
 'sink','refrigerator','blender','book','clock', 
 'vase','scissors','teddy bear','hair drier','toothbrush',   
 'hair brush' ]


engine = pyttsx3.init()

def look_up(search_number):
    
    return COCO_categories[int(search_number)-1]

def detect_callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    #print "data data" + data.data +"END"
    print "Object detected: " + look_up(data.data)
    engine.say(look_up(data.data))
    #this line is a problem when listening to 2 nodes
    engine.runAndWait()

def chatter_callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    engine.say(data.data) #what happens if you use data instead of data.data
    #this line is a problem when listening to 2 nodes
    engine.runAndWait()

def speak():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    rospy.init_node('speak', anonymous=True)

    rospy.Subscriber('chatter', String, chatter_callback)
    rospy.Subscriber('detected_objects', String, detect_callback)
    

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    #listener()
    speak()





