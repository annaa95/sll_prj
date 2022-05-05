# The Supervisor Controller
import numpy as np
import random
import csv
import pandas as pd

from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from collections.abc import Iterable
from std_msgs.msg import Float32MultiArray


OBSERVATION_SPACE = 13 # y, dx, rho
ACTION_SPACE = 3 # alphas, thetas, omegas


class SLL_Supervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()

		#Robot init
		self.super = self.supervisor.getFromDef("Supervisor")
		self.robot = None

		self.pressure_emitter = self.supervisor.getDevice("pressure_emitter")
		self.pressure_emitter.setChannel(1)

		self.touch_names = list()
		self.touch = list()	
		self.extra_obs = list()	
		#display init
		self.display = self.supervisor.getDevice("display")
		self.height = self.display.getHeight()
		self.half_height = self.height / 2
		self.width = self.display.getWidth()
		self.display_unit_x = (self.width)*self.timestep/ 20000
		self.display_unit_y = (self.height*0.5) / 700
		self.mem_length = int(self.width/self.display_unit_x)
		self.mem = np.zeros(self.mem_length)
		#self.reset_display()

		#state variables init
		self.y_min = 0.05
		self.vel_sgn_old = -1
		self.vel_old = 0.0

		self.medium = self.supervisor.getFromDef("MEDIUM")
		self.rho = self.medium.getField("density")

		self.count = 0
		self.t0 = self.supervisor.getTime()

			
		self.path = "/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager/objects/"
		self.respawnRobot() #create the robot
		

	def __del__(self):
		print('Destructor called.')

	##################### Robot Creation/Destruction ###################################
	def respawnRobot(self, atInit = True):			
				
		rootNode = self.supervisor.getRoot() 			# 1- Get the root children field
		rootChildren = rootNode.getField("children")	# 2- Get a list of all objects in the scene
		
		"""if self.robot is not None:					   	# 3- Check existance of robot
			self.robot.remove()
		"""
		if atInit: 
			#print("loading SLL_ROBOT")
			rootChildren.importMFNode(-2, self.path+"SLL_Robot.wbo") 	# 4- Add a node in the second to last position (ROBOT)
		else:
			#print("loading SLL_ROBOT_sensor")
			rootChildren.importMFNode(-2, self.path+"SLL_Robot_Sensors.wbo") 	# 4- Add a node in the second to last position (ROBOT)

		self.robot = self.supervisor.getFromDef("ROBOT")# 5- Get the new robot references
		
		self.robotNode = rootChildren.getMFNode(-2)			# 6- Get the root children field of the robot Node
		self.rbtSeg1 = self.robotNode.getField("children").getMFNode(-1).getFieldByIndex(2).getSFNode() # shank rf
		#print(self.rbtSeg1)
		self.rbtSeg2 = self.robotNode.getField("children").getMFNode(-1).getFieldByIndex(2).getSFNode().getField("children").getMFNode(-1).getField("children").getMFNode(0).getFieldByIndex(2).getSFNode()	
		self.pos_old = self.robotNode.getPosition()
		filename = self.path+"sensorSet.csv"						# 7- Load the information on sensors from a csv file
		
		if atInit:
			self.loadSensorsSet(filename, self.robotNode)	
		
		# 10- Reset the simulation physics to start over (affects the inertia of bodies)
		self.supervisor.simulationResetPhysics()

	def loadSensorsSet(self, filename, robotNode):		
		# get root to all segments in the robot tree
		robotChildren = robotNode.getField("children") 	#emitter receiver T0(body) HJ(hip)
		HJnode_hip= robotChildren.getMFNode(-1) 		#	HJ(hip)
		HJhip_ep = HJnode_hip.getFieldByIndex(2) 		#SFNode- solid. figli--> T1(coscia RF) T2(knee joint RF)
														# 	T1(coscia RF)
		ThighNode_rootchildren = HJhip_ep.getSFNode().getField("children").getMFNode(0).getField("children") # qui si può sensorizzare la coscia
														#	T2(knee joint RF)
		HJnode_knee = HJhip_ep.getSFNode().getField("children").getMFNode(-1).getField("children").getMFNode(0)		
		HJknee_ep = HJnode_knee.getFieldByIndex(2) 		#SFNode- solid. figli--> HJ(SEA)
		HJnode_sea = HJknee_ep.getSFNode().getField("children").getMFNode(0) 
		HJsea_ep = HJnode_sea.getFieldByIndex(2) 		#SFNode- solid. (tibia RF) 
		HJnode_SEArootchildren = HJsea_ep.getSFNode().getField("children") 		
		csvFile = pd.read_csv(filename, sep=';')
		
		for lines in range(0,len(csvFile.index)):
			#print(csvFile.loc[lines]['Link'])
			if csvFile.loc[lines]['Link'] == 'foot':
				HJnode_SEArootchildren.importMFNode(-1, self.path+"TouchSensor.wbo")	# Touch sensor has 21 fields
				TS_name = HJnode_SEArootchildren.getMFNode(-1).getField("name")
				TS_pos = HJnode_SEArootchildren.getMFNode(-1).getField("translation") 
				Pos = [0.05*csvFile.loc[lines]['Pos']-0.025, -0.25, 0]
				TS_type = HJnode_SEArootchildren.getMFNode(-1).getField("type")
				skip_enable = False
			elif csvFile.loc[lines]['Link'] == 'shank':
				HJnode_SEArootchildren.importMFNode(-1, self.path+"TouchSensor.wbo")
				TS_name = HJnode_SEArootchildren.getMFNode(-1).getField("name")
				TS_pos = HJnode_SEArootchildren.getMFNode(-1).getField("translation") 
				Pos = [-0.05, csvFile.loc[lines]['Pos']*0.5-0.25, 0]
				TS_type = HJnode_SEArootchildren.getMFNode(-1).getField("type")
				skip_enable = False
			elif csvFile.loc[lines]['Link'] == 'thigh':
				ThighNode_rootchildren.importMFNode(-1, self.path+"TouchSensor.wbo")
				TS_name = ThighNode_rootchildren.getMFNode(-1).getField("name")
				TS_pos = ThighNode_rootchildren.getMFNode(-1).getField("translation") 
				Pos = [-0.05, csvFile.loc[lines]['Pos']*0.5-0.25, 0]
				TS_type = ThighNode_rootchildren.getMFNode(-1).getField("type")
				skip_enable = False
			else:
				skip_enable = True

			if not(skip_enable):
				name = "TouchSensor_no"+str(csvFile.loc[lines]['FBG_ID'])
				TS_name.setSFString(name)
				TS_pos.setSFVec3f(Pos)
				TS_type.setSFString(csvFile.loc[lines]['Type']) # choose between bumper; force; force-3d
		# Send to the robot the information onn the number os sensors created
		
		string =self.robotNode.exportString()

		text_file = open(self.path+"SLL_Robot_Sensors.wbo", "w")
		text_file.write("#VRML_OBJ R2021b utf8\n"+string)
		text_file.close()	
		self.sensors = 	len(csvFile.index)-1
		#self.handle_emitter([len(csvFile.index)])

		
	def reset(self, ep):
		
		#print("resetting episode", ep)
		if ep ==0:
			pass
		else:
			self.respawnRobot(atInit=False)
		self.should_done = False
		self.message = None
		self.observation = np.zeros(OBSERVATION_SPACE)
		self.count = 0
		#self.reset_display()
		return self.observation

	######################### Communication Manager ###################################	
	def pres_emit_data(self, data):
		assert isinstance(data, Iterable), \
			"The action object should be Iterable"

		pres_msg = (",".join(map(str, data))).encode("utf-8")
		self.pressure_emitter.send(pres_msg)

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
			check = (self.vel_sgn_old==1 and self.vel_sgn==-1 and self.pos[1]>self.y_min)
		except:
			check = False
		if check :
			self.count +=1
			#self.plot_event(self.supervisor.getTime()-self.t0)
			#print("jump no. = "+str(self.count))
		
		self.vel_sgn_old = self.vel_sgn
		self.pos_old = self.pos
		self.vel_old = self.vel[1]
		
		return check
	######################### Auxiliar Functions : Sig Proces ###################################	
	def movAvFilt(self,y, lag=10):
		try:
			self.buffer[0:lag-2] = self.buffer[1:lag-1]
		except:
			self.buffer =0*np.ones(lag)

		self.buffer[lag-1] = y		
		y_filt = np.mean(self.buffer)

		#self.plot_display(self.supervisor.getTime()-self.t0, y_filt*100,0x08E720)
		return y_filt	

	def EstimDensity(self):
		vel = self.robotNode.getVelocity() #linear and angular in world coordinates
		dt = self.timestep/1000 #in seconds
		try:
			acc_t = (vel[1]-self.vel_tm1)/dt
		except :
			acc_t = (vel[1]-0)/dt
		self.vel_tm1 = vel[1]
		rho_w_currEp = (1-np.abs(acc_t)/9.81)*1.5e3
		self.rho = rho_w_currEp
		return rho_w_currEp
	######################### Agent Functions ###################################	

	def get_observations(self):

		# receive the message from the robot node
		# message will be built up as: 	
		message = self.handle_receiver()
		try:
			message = list(message)
			#message.insert(0, self.rho.getSFFloat()) #rho
			message.insert(0, self.EstimDensity()) #rho
			message.insert(0, self.robotNode.getPosition()[1])#y
			message.insert(0, self.robotNode.getVelocity()[0])#dx
		except:
			pass
		return message

	def get_reward(self, action=None):
		"""
		logica di attribuzione del reward
		tanto più alto tanto più angolo è prossimo a 90
		0 se ha fallito
		"""
		# as in Thuruthel, Picardi, Iida, Laschi, Calisti
		try:
			if self.pos[1] > self.y_min:
				R = np.log(1+1/np.abs(self.vel[0]+1e-10))
			else:
				R = 0
		except:
			R = 0
		return R
    
	def is_done(self):
		try:
			pos = self.robotNode.getPosition()
			vel = self.robotNode.getVelocity()[0]
			done =(pos[1]<self.y_min and self.pos_old[1]<self.y_min) #corpo hit the round e non si rialza
			#print(pos)
			done = done or self.count==5 or np.abs(vel)<0.01 #fattp 5 step o raggiunto obiettivo
		except:
			done = False
		return done
    
	def get_info(self):
		return super().get_info()

	def step(self, action, repeatSteps=	10000, iter_=0):
		"""
		Custom implementation of step function. !step = 1 hopping period from apex to apex
		:param action: Iterable that contains the action value(s)
		:type action: iterable
		:param repeatSteps: Number of steps to repeatedly do the same action before returning, defaults to 1
		:type repeatSteps: int, optional
		:return: observation, reward, done, info
		"""		
		action = list(action)
		action.insert(0, 'MFP') 	
		stp=0
		while not(self.apexEventDetected()) :
			self.supervisor.step(self.timestep)
			try:
				self.pres_emit_data([self.robotNode.getPosition()[1]])
			except:
				break
			self.handle_emitter(action)
			stp+=1
			if stp==repeatSteps:
				return(
					self.get_observations(),
					self.get_reward(),
					self.is_done(),
					self.get_info(),
				)
		return(
				self.get_observations(),
				self.get_reward(),
				self.is_done(),
				self.get_info(),
			)				
	######################### Display Mangement ###################################	
	
	def reset_display(self):
		self.display.setColor(0x000000) #black
		self.display.fillRectangle(0, 0, self.width - 1, self.height - 1)
		#self.supervisor.step(self.timestep)
		self.display.setColor(0xFF0000)
		self.display.drawLine(0, int(self.half_height), int(self.width), int(self.half_height))
		#self.supervisor.step(self.timestep)
		self.display.setColor(0xFFFFFF)

	def plot_display(self,x,y, color = 0xFFFFFF):
		self.display.setColor(color)
		current_t = (1000*x/self.timestep)*self.display_unit_x
		current_y = self.half_height-(y)*10*self.display_unit_y
		self.display.drawPixel(int(current_t), int(current_y))
		
	def plot_event(self, x):
		current_t = (1000*x/self.timestep)*self.display_unit_x	
		self.display.setColor(0xFF0000)
		self.display.drawLine(int(current_t), int(self.height), int(current_t), 0)
		self.display.setColor(0xFFFFFF)
		