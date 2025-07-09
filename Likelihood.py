import torch
import numpy as np
import torch.nn.functional as fucntional
from utils import device, double_unsqz
from Env import RestlessMultiArmedBandit
import pickle

def load_all_subj_df():
    file = open('Data/all_subj_df.pkl', 'rb')
    all_subj_df = pickle.load(file)
    file.close()
    return all_subj_df

def load_all_subj_overall_performance():
    file = open('Data/all_subj_overall_performance.pkl', 'rb')
    all_subj_overall_performance = pickle.load(file)
    file.close()
    return all_subj_overall_performance 

def test_subj_epsd(trained_model,state_list, cfg, verbose = False,return_all = False):
    
    hidden_states = (torch.zeros(1,1,cfg['hidden_size'],dtype=torch.float32, device = device ), torch.zeros(1,1,cfg['hidden_size'],dtype=torch.float32, device = device ))
    lstm_activations = np.zeros((len(state_list),cfg['hidden_size']))
    actions_probs = np.zeros((len(state_list), RestlessMultiArmedBandit.N_ARMS))
    critic_values = np.zeros(len(state_list))

    for t,state in enumerate(state_list):
        with torch.no_grad():
            lstm_output,hidden_states = trained_model.lstm(state, hidden_states)
            actions_log_prob  = trained_model.actor(lstm_output)
            if return_all:
                value_t = trained_model.critic(lstm_output)
                actions_probs_t = torch.exp(actions_log_prob)
        lstm_activations[t,:] = lstm_output
        if return_all:
            actions_probs[t,:] = actions_probs_t[0,0]
            critic_values[t] = value_t
    if return_all:
        return lstm_activations, actions_probs, critic_values
    return lstm_activations

def get_state_list(subj_actions,subj_rewards):
    state_list = []
    for t in range(len(subj_actions)):
        t_reward = torch.tensor([subj_rewards[t]], dtype=torch.float32, device = device)
        one_hot_action = fucntional.one_hot(torch.LongTensor([subj_actions[t]], device = device),num_classes=RestlessMultiArmedBandit.N_ARMS).squeeze()
        t_trial =torch.tensor([t], dtype=torch.float32, device = device)
        state_list.append(double_unsqz(torch.cat([t_trial,t_reward, one_hot_action],0)))
    return state_list

def calc_likelihood(subj_actions, subj_fb, trained_model, cfg, N_iter = 10):
    state_list = get_state_list(subj_actions,subj_fb)
    _, actions_probs, _ = test_subj_epsd(trained_model,state_list, cfg, return_all=True)
    s_all_iter = np.zeros(N_iter)
    for n in range(N_iter):
        chosen_actions = np.asarray([torch.multinomial(torch.Tensor(actions_probs[i]), 1).numpy() for i in range(len(actions_probs))]).squeeze()
        s = 0
        for i in range(len(subj_actions)-1):
            if subj_actions[i+1] == chosen_actions[i]:
                s+=1
        s_all_iter[n] = s/len(subj_actions)     
    likelihood = np.mean(s_all_iter)
    return likelihood