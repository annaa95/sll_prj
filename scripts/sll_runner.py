#!/usr/bin/env python
# The learnign algorithm runner.
import sys
from supervisorManager.supervisorROSwrapper import SupervisorsControllerROSWrapper
from supervisorManager.networks_torch import DDPG
from supervisorManager.utils import plot_learning_curve

import numpy as np

OBSERVATION_SPACE = 13 # dx, y, rho, contacts 
ACTION_SPACE = 3 # alphas, thetas, omegas
TEST = False
ALPHA_RANGE = 120
OMEGA_RANGE = 360.0
THETA_RANGE = 120.0
ALPHA_MIN = -60.0 
OMEGA_MIN = 0.1
THETA_MIN = 30

class DDPG_runner(object):
    def __init__(self):
        """
        Initialize the supervisor
        """
        self.env = SupervisorsControllerROSWrapper()

        self.agent = DDPG(lr_actor=1.5,
                    lr_critic=1.5,
                    input_dims=[OBSERVATION_SPACE],
                    gamma=0.5,
                    tau=0.01,# tau <<1 -> 0.01 or smaller 
                    env=self.env,
                    batch_size=64,
                    layer1_size=400,
                    layer2_size=300,
                    n_actions=ACTION_SPACE,
                    load_models=True,
                    save_dir='/home/anna/catkin_ws/src/sll_extern_roscontroller/scripts/supervisorManager/models/PhilTabor/saved/ddpg/')
        
        self.score_history = []
        # Run outer loop until the episodes limit is reached
        np.random.seed(0)
        self.maxEpisodeNum = 100
        self.filename = 'SingleLeg_plot'
        self.figure_file = './models/PhilTabor/plot/' + self.filename + '.png'
        self.best_score = 0 #minimum of reward range

    def run(self):
        for ep in range(self.maxEpisodeNum):
            print("Episode: ", ep)
            #initialize a random process N for action exploration 
            # -> not very critical for success of the algorithm
            #receive initial observation state s1
            obs = list(map(float, self.env.reset(ep))) # la prima osservazione è sempre 0
            done = False
            score = 0
            while not done:
                act = self.agent.choose_action_train(obs).tolist()#range    [0-1]
                act = np.multiply(act,np.array([ALPHA_RANGE, OMEGA_RANGE, THETA_RANGE]))+np.array([ALPHA_MIN, OMEGA_MIN, THETA_MIN])
                new_state, reward, done, info = self.env.step(act)
                self.agent.remember(obs, act, reward, new_state, int(done))
                self.agent.learn()
                score += reward                 
                obs = list(map(float, new_state))
            print(done)    
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.agent.save_models()
            #self.env.reset(ep)"""

            self.env.supervisor.step(self.env.timestep) #bisogna dare tempo al restarting
            try:
                self.env.launch.shutdown()
            except:
                pass
            self.env.supervisor.simulationReset()
            self.env.supervisor.step(self.env.timestep) #bisogna dare tempo al restarting
                    
            print("===== Episode", ep, "score %.2f" % score,
                    "100 game average %.2f" % avg_score)
            
        x = [i+1 for i in range(self.maxEpisodeNum)]
        return x, self. score_history, self.figure_file

runner = DDPG_runner()
[x, score_history, figure_file] =runner.run()
plot_learning_curve(x, score_history, figure_file)    


