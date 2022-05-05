import gym
import numpy as np
from networks_torch import DDPG
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    agent = DDPG(lr_actor=0.0001, lr_critic=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001, env =env,
                    batch_size=64, layer1_size=400, layer2_size=300, 
                    n_actions=env.action_space.shape[0])
    np.random.seed(0)
    n_games = 3
    filename = 'Pendulum_plot'
    figure_file = './models/PhilTabor/plot/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action_train(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)



