
# Caution for lab or test environments only
## as of February 2020
## Python 2 (obsolete) , ROS Melodic Morenia (current) , Tensorflow 1.3 (obsolete) are used
## Pretrained Classifier required Tensorflow version 1.3
## and ROS Melodic problematic with Python 3
## new ROS availble summer 2020 for Python 3 and Tensorflow 2.X
## new pretrained classifer may be required for new ROS/Python/Tensorflow combination

Useful Commands

ROScore : This is a program that must be running in the background the whole time you are running the ROS system.  Open a terminal and type "roscore" and just let it run.

source devel/setup.bash : This command need to be typed into every new terminal to ensure the ROS in that terminal can see everything it need to run.

Catkin_make : This is similar in concept to CMAKE. This command is used after changes are made and you need to "recompile" the project.

ROSrun your_script : To execute a script or node in ROS type ROSRUN my_script.py 


##To Run as provided in Virtual Machine:
Open Visual Studio code all relevent files are in catkin_ws/src/beginner_tutorials/script
Open a terminal in Visual Studio code or outside it, run type ROSCORE and leave running
Open a second terminal type source devel/setup.bash, press enter then type catkin_make type enter. Then type rosrun beginner_tutorials speak.py and press enter and leave script running
Open a third terminal and type source devel/setup.bash, press enter then type catkin_make type enter. Then type rosrun beginner_tutorials finder.py and press enter and leave script running

finder.py should randomly select a picture and process it displaying Befopre and After images and sending a message 
speak.py should display a message and speak aloud the object eg "apple" , "orange" etc

program will run until CTRL-C is pressed in each terminal

Extra resources in scripts folder and also in Firefox browser look at source code comments for suggestions, links etc

Beware: if you make a new file it must be set to allow to execute file as a program in file permissions. You can adjust this in termianl or graphic file manager under properties. New node files need this done or they will not run.

Also select save from menu before running to make sure changes are updated.