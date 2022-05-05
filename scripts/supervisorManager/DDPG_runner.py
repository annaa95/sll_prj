
# The learnign algorithm runner.
from SupervisorController import SLL_Supervisor
from SLL_KeyboardControl import SLL_KeyboardController
#from SLL_TensorBoard import SLL_TensorboardLogger
from models.PhilTabor.networks_torch import DDPG
from models.PhilTabor.utils import plot_learning_curve

import numpy as np

OBSERVATION_SPACE = 12 # dx, y, rho, contacts 
ACTION_SPACE = 3 # alphas, thetas, omegas
TEST = False
ALPHA_RANGE = 120
OMEGA_RANGE = 360.0
THETA_RANGE = 140.0
ALPHA_MIN = 30.0 
OMEGA_MIN = 150.0
THETA_MIN = 29

def run():
    """
    Initialize the supervisor
    whenever we want to access attributes etc... from the supervisor controller
    we use the supervisorPre.
    """

    #supervisorPre = SLL_Supervisor()
    supervisorEnv = SLL_Supervisor()
    # gym-like enviroment
    #supervisorEnv = SLL_KeyboardController(supervisorPre)

    agent = DDPG(lr_actor=0.0001,
                lr_critic=0.,
                input_dims=[OBSERVATION_SPACE],
                gamma=0.5,
                tau=0.01,# tau <<1 -> 0.01 or smaller 
                env=supervisorEnv,
                batch_size=64,
                layer1_size=400,
                layer2_size=300,
                n_actions=ACTION_SPACE,
                load_models=False,
                save_dir='./models/PhilTabor/saved/ddpg/')
	
    score_history = []
    # Run outer loop until the episodes limit is reached
    np.random.seed(0)
    maxEpisodeNum = 15
    filename = 'SingleLeg_plot'
    figure_file = './models/PhilTabor/plot/' + filename + '.png'
    best_score = 0 #minimum of reward range
    score_history = []
    #supervisorEnv.supervisor.movieStartRecording(file= "/home/anna/Video/SingleLeg_learning.mp4", width=640, height=480, quality=100,codec=0, acceleration=1, caption=False)
    for ep in range(maxEpisodeNum):
        print("Episode: ", ep)
        #initialize a random process N for action exploration 
        # -> not very critical for success of the algorithm
        #receive initial observation state s1
        obs = list(map(float, supervisorEnv.reset(ep)))
        done = False
        score = 0
        while not done:
            act = agent.choose_action_train(obs).tolist()#range    [0-1]
            act = np.multiply(act,np.array([ALPHA_RANGE, OMEGA_RANGE, THETA_RANGE]))+np.array([ALPHA_MIN, OMEGA_MIN, THETA_MIN])
            new_state, reward, done, info = supervisorEnv.step(act, 100)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = list(map(float, new_state))
        print(done)    
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        supervisorEnv.reset(ep)
        supervisorEnv.supervisor.simulationReset()
        supervisorEnv.supervisor.step(supervisorEnv.timestep) #bisogna dare tempo al restarting
                
        print("===== Episode", ep, "score %.2f" % score,
                "100 game average %.2f" % avg_score)
    
    #supervisorEnv.supervisor.movieStopRecording()
   
    x = [i+1 for i in range(maxEpisodeNum)]
    plot_learning_curve(x, score_history, figure_file)    
