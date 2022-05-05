#!/usr/bin/env python
from std_msgs.msg import Float32MultiArray, Float32, Bool
import rospy 
import roslaunch
import rosservice
from sll_extern_roscontroller.srv import *

import numpy as np
import os
import sys
sys.path.insert(1, '/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager')
from SupervisorController import SLL_Supervisor
sys.path.insert(1, '/home/anna/catkin_ws/src/sll_extern_roscontroller/')
#writing a a supervisor controller subclass
# it will be a SLL_Supervisors object, i.e. inherithing everything from the SLL_Supervisor class

class SupervisorsControllerROSWrapper(SLL_Supervisor):
    def __init__(self):
        super().__init__()        
        s = rospy.Service('supervisor_node/deliverAction', deliverNewAction, self.handle_deliverActionSrv)
        print('Initializing ROS: connecting to ' + os.environ['ROS_MASTER_URI'])
        rospy.init_node('supervisor_node', anonymous=True)

        self.curr_stateVar_pub =rospy.Publisher("state_variable_status",Float32MultiArray, queue_size=10)
        self.events_pub =rospy.Publisher("events",Float32, queue_size=10)
        self.sars_pub = rospy.Publisher("sars_", Float32MultiArray, queue_size = 10)
        
        self.var = Float32MultiArray()
        self.ev = Float32()
        self.sars = Float32MultiArray()

        rospy.Subscriber("TD_no", Bool, self.handle_TD)

        self.last_stance_count = 0
        self.TD = False
        self.stance = 0

    def respawnRobot(self, atInit = True):	
        # Calling the parent's class method
        super().respawnRobot(atInit)
        #extending with its own functionality	
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/anna/catkin_ws/src/sll_extern_roscontroller/launch/sll_robot.launch"])
        self.launch.start()
        #script = launch.launch(node)
        rospy.loginfo("started")
        self.ret1_old =[0.0]*10
        self.ret1_old.insert(0, self.EstimDensity()) #rho
        self.ret1_old.insert(0, self.robotNode.getPosition()[1])
        self.ret1_old.insert(0, self.robotNode.getVelocity()[0])#dx
        self.rho = self.EstimDensity()
        #rospy.sleep(3)
        
    def handle_supplyCntcClnt(self):
        rospy.wait_for_service('robot_node/supplyCntc')
        try:
            handlefunc = rospy.ServiceProxy('robot_node/supplyCntc', supplyFirstCntc)
            resp = handlefunc(True)
            message= list(resp.timeCntc)
            #print(message)
        except rospy.ServiceException as e:
            message = None
            #print("Service call failed: %s"%e)       
        return message

    def handle_deliverActionSrv(self, req):
        """
        This function will be used instead of the handle_emitter(action) function
        The output is  hlb == 'MFP', alpha_td, omega, theta_0, 
        """
        if req.deliver == True and self.toDeliver:
            return deliverNewActionResponse("MFP", self.action[0], self.action[1], self.action[2])
    
    def handle_TD(self, data):
        self.TD_old = self.TD
        self.TD = data.data
        if self.TD_old is False and self.TD is True:
            self.stance +=1

    def publish_cur_stateVar(self, event=None):
        try:
            self.extra_obs =[self.supervisor.getTime(), 
                            self.rho]
            pos = self.robotNode.getPosition()
            vel = self.robotNode.getVelocity()               
            rot = self.robotNode.getOrientation() #[ R[0] R[1] R[2] ];[ R[3] R[4] R[5] ];[ R[6] R[7] R[8] ]
            femPos =self.rbtSeg1.getField("translation").getSFVec3f()
            femRot = self.rbtSeg1.getField("rotation").getSFRotation()
            tibPos = self.rbtSeg2.getField("translation").getSFVec3f()
            tibRot = self.rbtSeg2.getField("rotation").getSFRotation()

            for i in range(2):
                self.extra_obs.append(pos[i]) #esclusa z 
            for i in [0, 1, 5]:
                self.extra_obs.append(vel[i]) #esclusa linear in z and angular in x and y
            for i in range(9):
                self.extra_obs.append(rot[i])
            
            for i in range(2):
                self.extra_obs.append(femPos[i]) #esclusa z    
            for i in range(4):
                self.extra_obs.append(femRot[i])   
                                  
            for i in range(2):
                self.extra_obs.append(tibPos[i]) #esclusa z    
            for i in range(4):
                self.extra_obs.append(tibRot[i])  
            
            self.var.data = self.extra_obs

        except:
            self.var.data = [0.0]*27
            self.var.data.insert(0, self.supervisor.getTime())

        self.curr_stateVar_pub.publish(self.var)	

    def publish_event(self, event=None):
        self.ev.data = self.supervisor.getTime()
        self.events_pub.publish(self.ev)      
    
    def publish_sars(self, event= None):
        ret1 = self.get_observations()
        ret2 = self.get_reward()
        ret3 = self.is_done()
        self.sars.data = self.ret1_old
        for i in range(len(self.action)):
            self.sars.data.append(self.action[i])
        self.sars.data.append(ret2)
        for i in range(len(ret1)):
            self.sars.data.append(ret1[i])
        self.sars.data.append(float(ret3))
        self.ret1_old = ret1
        self.sars_pub.publish(self.sars)      
        return(ret1,
               ret2,
               ret3,
               self.get_info())
	################          overwritten function from parent        #############################
    
    def apexEventDetected(self):
        """
        This function outputs a boolean, check, which detects 
        sign change of the vertical velocity (from upward to downward) above a security treshold
        note:  it may not corretly deal with slow crossing such as sign(t)= +1 0 -1
        """
        try:
            self.pos = self.robotNode.getPosition()
            self.vel = self.robotNode.getVelocity() #linear and angular in world coordinates
            vel_filt = self.movAvFilt(self.vel[1])
            self.vel_sgn = np.sign(vel_filt)
            check = (self.vel_sgn_old==1 and self.vel_sgn==-1 and not(self.last_stance_count==self.stance))
        except:
            check = False
        if check :
            self.count +=1
            self.last_stance_count = self.stance
            #self.plot_event(self.supervisor.getTime()-self.t0)
                    
        self.vel_sgn_old = self.vel_sgn
        self.pos_old = self.pos
        self.vel_old = self.vel[1]
        
        return check    
        

    def get_observations(self):

        # receive the message from the robot node
        # message will be built up as: 	
        message = self.handle_supplyCntcClnt()
        message.insert(0, self.EstimDensity()) #rho
        message.insert(0, self.robotNode.getPosition()[1])#y
        vel =self.robotNode.getVelocity()
        message.insert(0, vel[0])#dx
        message.insert(0, vel[5])#dphi_z
        return message	
    
    
    def step(self, action, repeatSteps=	10000):
        """
        Custom implementation of step function. !step = 1 hopping period from apex to apex
        :param action: Iterable that contains the action value(s)
        :type action: iterable
        :param repeatSteps: Number of steps to repeatedly do the same action before returning, defaults to 1
        :type repeatSteps: int, optional
        :return: observation, reward, done, info
        """		
        self.toDeliver = True
        self.publish_event()
        self.action = list(action)
        stp=0
        while not(self.apexEventDetected()) and not rospy.is_shutdown():#
            #self.rho.setSFFloat(float(1e3+np.random.randn(1)*10))  			
            self.supervisor.step(self.timestep)
            if stp > 1:
                self.toDeliver = False
            self.publish_cur_stateVar()
            stp+=1
            if stp==repeatSteps:
                print("Reached max step num")
                self.publish_sars()
                return(
                self.publish_sars()
                )
        return(
        self.publish_sars()
        )
