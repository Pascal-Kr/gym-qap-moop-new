import gym
from gym_qap_moop_new.Agents.Train_Agent import DQNAgent
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import statistics

current_time = datetime.now().strftime('%b%d_%H-%M')
writer = SummaryWriter(log_dir=os.path.join('logs_pytorch_Test', current_time + '_Rücklauf100_1k'))


if __name__ == '__main__':
    env = gym.make('QAPMOOP-v0')
    ename='Noise100'
    Aggregate_Stats_Every=50
    Test_runs = 200
    MHCmin, MHCmax, Returnflowmin, Returnflowmax, Noisemin, Noisemax, Noise_scoremin, Noise_scoremax = env.Test_run(Test_runs)
    average_reward=0
    load_checkpoint = True
    write_Logfile = False
    Show_Actions = False
    Build_Average = False
    Aggregate_Stats_Every = 10
    episodes=5000
    if write_Logfile == True:
        episodes = 1000
    
    # For stats
    ep_rewards = []
    ep_Minimums = []
    ep_last_Costs = []
    ep_last_Returnflow = []
    ep_last_Noise_score = []
    Relativ_changes_Noise_score = []
    Relativ_changes_MHC = []
    Relativ_changes_Returnflow = []


    agent = DQNAgent(episodes, gamma=0.8, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space_values),
                     n_actions=env.action_space.n, Replay_Memory_Size=5000, eps_min=0,
                     batch_size=64, Update_Target_Every=1000,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name=ename)
    

    if load_checkpoint:
        agent.load_models()
        agent.epsilon=0


    for episode in range(1, episodes + 1):
        done = False
        current_state = env.reset()
        if Show_Actions == True:
            env.render()
        if write_Logfile == False:
            print('Initial Noise Score: ' + str(env.initial_Noise_score))
            print('Initial MHC: ' + str(int(env.initial_MHC)))
            print('Initial Return Flow: ' + str(env.initial_Returnflow))
        episode_reward = 0
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            if Show_Actions == True:
                env.render()
            if write_Logfile == False:
                print(env.actions[action])
                print('Noise Score: ' + str(info['Noise']))
                print('MHC: ' + str(int(info['MHC'])))
                print('Return Flow: ' + str(info['Return Flow']))
                print('MHC Reward: ' + str(env.MHCreward))
                print('Return Flow reward: ' + str(env.Returnflowreward))
                print('Noise Reward Intervals: ' + str(env.Noiserewardintervals))
                print('Reward for action: ' + str(reward))
                print('Episoden Reward: ' + str(episode_reward))
                print('')
            
            current_state = new_state
        print('')
        Relativ_change_Noise = round((env.initial_Noise_score-info['Noise'])/info['Noise']*100,2)
        Relativ_change_MHC = round((env.initial_MHC-info['MHC'])/info['MHC']*100,2)
        Relativ_change_Returnflow = round((env.initial_Returnflow-info['Return Flow'])/info['Return Flow']*100,2)
        Relativ_changes_Noise_score.append(Relativ_change_Noise)
        Relativ_changes_MHC.append(Relativ_change_MHC)
        Relativ_changes_Returnflow.append(Relativ_change_Returnflow)
        print('Relative change between Start and End (Noise): ' + str(Relativ_change_Noise) + '%')
        print('Relative change between Start and End (MHC): ' + str(Relativ_change_MHC) + '%')
        print('Relative change between Start and End (Returnflow): ' + str(Relativ_change_Returnflow) + '%')
        print('')
        ep_rewards.append(episode_reward)
        ep_last_Costs.append(info['MHC'])
        ep_last_Returnflow.append(info['Return Flow'])
        ep_last_Noise_score.append(info['Noise'])
        if write_Logfile == True:
            writer.add_scalar('Episode Reward', episode_reward, episode)
            writer.add_scalar('Last MHC', info['MHC'], episode)
            writer.add_scalar('Last Return Flow', info['Return Flow'],episode)
            writer.add_scalar('Last Noise Score', info['Noise'],episode)
        if not episode % Aggregate_Stats_Every and Build_Average == True:
            average_reward = sum(ep_rewards[-Aggregate_Stats_Every:])/len(ep_rewards[-Aggregate_Stats_Every:])
            writer.add_scalar('Average Reward', average_reward, episode)
            
Minimum_MHC = min(ep_last_Costs)          
Maximum_MHC = max(ep_last_Costs)
average_MHC = int(statistics.mean(ep_last_Costs))
average_changes_MHC = statistics.mean(Relativ_changes_MHC)
stand_deviation_MHC = np.std(ep_last_Costs)
Variation_coeff_MHC = np.std(ep_last_Costs) / np.mean(ep_last_Costs) *100
Minimum_Returnflow = min(ep_last_Returnflow)
Maximum_Returnflow = max(ep_last_Returnflow)
average_Returnflow = int(statistics.mean(ep_last_Returnflow))
average_changes_Returnflow = statistics.mean(Relativ_changes_Returnflow)
stand_deviation_Returnflow = np.std(ep_last_Returnflow)
Variation_coeff_Returnflow = np.std(ep_last_Returnflow) / np.mean(ep_last_Returnflow) *100
Minimum_Noise_score = min(ep_last_Noise_score)
Maximum_Noise_score = max(ep_last_Noise_score)
average_Noise_score = int(statistics.mean(ep_last_Noise_score))
average_changes_Noise_score = statistics.mean(Relativ_changes_Noise_score)
stand_deviation_Noise_score = np.std(ep_last_Noise_score)
Variation_coeff_Noise_score = np.std(ep_last_Noise_score) / np.mean(ep_last_Noise_score) *100
average_reward = statistics.mean(ep_rewards)
            
            
