#!/usr/bin/env python
# The Robot Controller
from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
from std_msgs.msg import Float32MultiArray, Bool
# specify the module that needs to be 
# imported relative to the path of the 
# module
import math
import rospy 
import sys
from sll_extern_roscontroller.srv import *

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/anna/Documenti/webots_projects/SingleLegLearner_extern/controllers/robotController')

from robotController import SLL_RobotController

#from utilities import normalizeToRange
#from utilities import real_time_peak_detection
#from utilities import createFilt
#from tensorboardX import SummaryWriter
class RobotControllerROSWrapper:
    def __init__(self):
        self.message = None
        self.robotROS = SLL_RobotController()
        s = rospy.Service('robot_node/supplyCntc', supplyFirstCntc, self.handle_supplyCntcSrv)
        
        self.curr_touchSens_pub =rospy.Publisher("touch_sensors_status",Float32MultiArray, queue_size=10)
        self.curr_TD_pub =rospy.Publisher("TD_no",Bool, queue_size=10)
        self.jointState_pub =rospy.Publisher("joint_state",Float32MultiArray, queue_size=10)
        self.imu_pub =rospy.Publisher("imu_state",Float32MultiArray, queue_size=10)
        
        rospy.Subscriber("state_variable_status", Float32MultiArray, self.handle_bodyPos)

        self.TD_no = Bool()
        self.TD_no.data = False
        self.sens = Float32MultiArray()
        self.imu = Float32MultiArray()
        self.motorState = Float32MultiArray()

        rospy.Timer(rospy.Duration(1.0/100.0), self.publish_cur_imuSens)        
        rospy.Timer(rospy.Duration(1.0/100.0), self.publish_cur_touchSens)
        rospy.Timer(rospy.Duration(1.0/100.0), self.publish_cur_motorStatus)
    
    def handle_supplyCntcSrv(self, req):
        """
        This function will be used instead of the handle_emitter(action) function
        The output is  hlb == 'MFP', alpha_td, omega, theta_0, 
        """
        if req.send == True:
            return supplyFirstCntcResponse(self.time_first_contact)
        
    def handle_deliverActionClnt(self):
        rospy.wait_for_service('supervisor_node/deliverAction')
        try:
            handlefunc = rospy.ServiceProxy('supervisor_node/deliverAction', deliverNewAction)
            resp = handlefunc(True)
            #print('alpha_received:',resp.alpha)
            self.action = [resp.highLevelBehavior, resp.alpha, resp.omega, resp.theta0]
            #self.time_first_contact = [float('nan')] * len(self.robotROS.TouchSensors)
            self.time_first_contact = [0.0] * len(self.robotROS.TouchSensors)
            self.time_from_last_action = self.robotROS.robot.getTime()
        except rospy.ServiceException as e:
            pass
            #print("Service call failed: %s"%e)       
        self.robotROS.use_message_data(self.action)

    def handle_bodyPos(self,data):
        headPos = data.data[3]
        self.robotROS.headHitGround= headPos<0.01

    def publish_cur_motorStatus(self, event=None):
        self.motorState.data = [self.robotROS.robot.getTime(), self.robotROS.encoders[0].getValue(), self.robotROS.encoders[1].getValue(), self.robotROS.motor[0].getVelocity(), self.robotROS.motor[1].getVelocity(), self.robotROS.motor[0].getTorqueFeedback(), self.robotROS.motor[1].getTorqueFeedback()]
        self.jointState_pub.publish(self.motorState)  
        
    def publish_cur_imuSens(self, event=None):
        self.imu.data = [self.robotROS.robot.getTime()]
        for i in range(3):
            self.imu.data.append(self.robotROS.accelerometer.getValues()[i])
        for i in range(3):
            self.imu.data.append(self.robotROS.gyroscope.getValues()[i])
        self.imu_pub.publish(self.imu)  
    
    def publish_cur_touchSens(self, event=None):
        if self.message is not None:
            self.sens.data =self.message
            if len(self.sens.data)==len(self.robotROS.TouchSensors):
                self.sens.data.insert(0,self.robotROS.robot.getTime())
            self.curr_touchSens_pub.publish(self.sens)
    
    def read_contact_sensors(self):
        TD = []
        for i in range(len(self.robotROS.TouchSensors)):
            TD.append(self.robotROS.TouchSensors[i].getValue())
            if TD[i]> 0.0 and self.time_first_contact[i]==0.0:
                self.time_first_contact[i] = self.robotROS.robot.getTime()-self.time_from_last_action
        self.message = TD 
        self.robotROS.contacts = TD
        self.curr_TD_pub.publish(self.robotROS.contact)
        return self.message

    def run(self):
        self.robotROS.run()
        while self.robotROS.robot.step(self.robotROS.timestep) != 1:
            #print("running controller")
            #self.robotROS.handle_receiver()
            self.handle_deliverActionClnt()
            #self.robotROS.handle_emitter() #turn off for open loop behaviors
            self.read_contact_sensors()

if __name__ == "__main__":
    rospy.init_node("robot_node")
    robot_wrapper = RobotControllerROSWrapper()
    robot_wrapper.run()
    rospy.loginfo("ROS wrapper has been initialized")
