import torch

import gymnasium as gym
import os
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal

import numpy as np
import random as rd
import math

# Profiler
import cProfile
import re
import wandb

torch.autograd.set_detect_anomaly(True)

class ValueFunction(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, critic_learning_rate):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.critic_learning_rate = critic_learning_rate
        
        self.critic = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.critic_learning_rate)


    def forward(self, in_):
        if in_.dim() == 1:
            in_ = in_.unsqueeze(0)
        out = self.critic(in_)
        return out
    
class Policy(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, actor_learning_rate, std, device):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.actor_learning_rate = actor_learning_rate
        self.std = std
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.actor_learning_rate)

    def forward(self, in_):

        if in_.dim() == 1:
            in_ = in_.unsqueeze(0)
        h = self.actor(in_)

        epsilon = torch.randn(h.size(0), h.size(1)).to(self.device)
        z = h + self.std * epsilon
        return z, h, self.std
    
    def get_log_probability(self, z, mu, std):

        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(z).sum(dim=-1)

        return log_prob
    
class HalfCheetahBigLegEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        xml_path = os.path.join(
            os.path.dirname(__file__),
            "half_cheetah_bigleg.xml"
        )
        super().__init__(xml_file=xml_path, **kwargs)

class PPO(nn.Module):
    def __init__(self, epochs, training_iterations, batch_size, trajectory_length, 
                 n_actors, env1, env2, in_features, out_features, hidden_features, device, 
                 actor_learning_rate, critic_learning_rate, gamma, lambda_, epsilon, 
                 std, beta, d_targ, mode, n_nets, omega, omega12, run, toggle_log, alternating_step):
        super().__init__()
        
        self.epochs = epochs
        self.training_iterations = training_iterations
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.n_actors = n_actors
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.beta = beta
        self.omega = omega
        self.omega12 = omega12
        self.d_targ = d_targ
        self.env1 = env1
        self.env2 = env2
        self.flag_env = 1
        self.device = device
        self.std = std
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.mode = mode
        self.n_nets = n_nets
        self.loss_coeff = beta * (omega ** torch.arange(0, self.n_nets)).to(device)
        self.run = run
        self.toggle_log = toggle_log
        self.alternating_step = alternating_step

        if n_nets > 1:
            base_actor = Policy(
                            in_features,
                            out_features,
                            hidden_features,
                            actor_learning_rate,
                            std,
                            device
                        ).to(self.device)

            self.actor_list = nn.ModuleList()

            for i in range(self.n_nets):
                actor = Policy(
                    in_features,
                    out_features,
                    hidden_features,
                    (self.omega**(-i)) * actor_learning_rate,
                    std,
                    device
                ).to(self.device)
                # the actors starts all with the same weights
                actor.load_state_dict(base_actor.state_dict())
                self.actor_list.append(actor)

        else:
            self.actor = Policy(in_features, out_features, hidden_features, actor_learning_rate, std, device)
        
        self.critic = ValueFunction(in_features, out_features, hidden_features, critic_learning_rate)

    def switch_env(self):
        print("Swapping task...")
        self.flag_env = (self.flag_env+1) % 2
        if self.flag_env == 0:
            return self.env1
        return self.env2

    def train_manager(self):
        if self.mode == 'pc':
            self.train_model_pc()
        elif self.mode in ['clip', 'kl_fixed', 'kl_adaptive']:
            self.train_model()
        else:
            print("[train_manager] method not allowed")
    
    def train_model(self):

        N = self.n_actors  #number of actors
        T = self.trajectory_length # trajectory length
        
        for i in range(self.training_iterations):
            dataset = []
            print(f"[train]: starting dataset creation at iteration n {i}")

            # decaying std
            # self.std = self.std - 0.025*self.std
            # self.run.log({"std": self.std})

            if self.alternating_step > 0:
                if (i % self.alternating_step) == 0:
                    # env = self.switch_env()
                    self.std = 1
            else:
                env = self.env1

            with torch.no_grad():

                # self.eval_model(env)
                adv_list = []
                cum_reward = 0
                for _ in range(N): #for each actor

                    # initialize first state
                    s_prime, _ = env.reset()
                    s_prime = torch.tensor(s_prime, dtype=torch.float32).to(self.device)
                    
                    trajectory = []

                    for t in range(T):

                        action, mu, std = self.actor(s_prime)
                        log_policy = self.actor.get_log_probability(action, mu, std)

                        s, reward, terminated, _, _ = env.step(action.squeeze(0).cpu().detach().numpy())
                        s = torch.tensor(s, dtype=torch.float32).to(self.device)
                        reward = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
                        s_prime = s_prime.unsqueeze(0)
                        trajectory.append([s_prime, action, reward, log_policy])
                        s_prime = s
                        cum_reward += reward

                        if terminated:
                            break

                    dynamic_target = 0 if terminated else self.critic(s)
                    for t in range(len(trajectory)-1, -1, -1): #I want the range from [T-1 to 0]
                        
                        dynamic_target = trajectory[t][2] + self.gamma*dynamic_target #taking the reward
                        advantage = dynamic_target - self.critic(trajectory[t][0])
                        trajectory[t] = tuple(trajectory[t] + [dynamic_target.unsqueeze(0), advantage.unsqueeze(0)])

                        dataset.append(trajectory[t])
                        adv_list.append(advantage)

                adv_std, adv_mean = torch.std_mean(torch.tensor(adv_list))
                print(f"[training]: avg cum reward {cum_reward / N}")
                if self.toggle_log:
                    self.run.log({
                        "avg cum_reward_training": cum_reward / N,
                        "adv mean": adv_mean,
                        "adv std": adv_std,
                        })
            
            print(f"[training]: ending dataset creation with dataset size {len(dataset)}")

            self.actor.zero_grad()
            self.critic.zero_grad()
            # Starts the training process

            for e in range(self.epochs):
                
                #print(f"[train]: epoch n {e}")
                avg_loss_value = 0
                avg_loss_ppo = 0
                rd.shuffle(dataset) #shuffle in-place
                
                assert(self.batch_size <= len(dataset))

                for mini_idx in range(0, len(dataset), self.batch_size):
                    
                    # form mini_batch
                    mini_batch = dataset[mini_idx: mini_idx+self.batch_size]

                    state_mini = torch.stack(list(map(lambda elem: elem[0].squeeze(), mini_batch)))
                    action_mini = torch.stack(list(map(lambda elem: elem[1].squeeze(), mini_batch)))
                    log_policy_mini = torch.stack(list(map(lambda elem: elem[3].squeeze(), mini_batch)))
                    target_mini = torch.stack(list(map(lambda elem: elem[4].squeeze(), mini_batch)))
                    advantage_mini = torch.stack(list(map(lambda elem: elem[5].squeeze(), mini_batch)))
                    
                    # Normalize advantage_mini
                    advantage_mini = ((advantage_mini-adv_mean) / (adv_std+1e-8))

                    _, mu_mini, std_mini = self.actor(state_mini) # std is a scalar!
                    new_log_policy_mini = self.actor.get_log_probability(action_mini, mu_mini, std_mini)   

                    new_value_mini = self.critic(state_mini)
                    
                    self.actor.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    
                    if self.mode == 'clip':
                        loss_ppo = self.loss_clip(new_log_policy_mini, log_policy_mini, advantage_mini)
                    elif (self.mode == 'kl_fixed') or (self.mode == 'kl_adaptive'):
                        loss_ppo = self.loss_kl_standard(new_log_policy_mini, log_policy_mini, advantage_mini)

                    loss_value = self.loss_value(new_value_mini, target_mini)

                    avg_loss_ppo += loss_ppo
                    avg_loss_value += loss_value

                    loss_ppo.backward()
                    loss_value.backward()

                    self.actor.optimizer.step()
                    self.critic.optimizer.step()

                total_minibatch = math.floor(len(dataset) // self.batch_size)
                print(f"epoch {e} -> [avg actor loss]: {avg_loss_ppo / total_minibatch}  [critic loss]: {avg_loss_value / total_minibatch}")
                if self.toggle_log:
                    self.run.log({
                            "avg_loss_ppo": avg_loss_ppo / total_minibatch,
                            "avg_loss_value": avg_loss_value / total_minibatch,
                        })

            #self.save_parameters("partial_models/model"+str(i)+".pt")

    def train_model_pc(self):

        N = self.n_actors  #number of actors
        T = self.trajectory_length # trajectory length
        
        for i in range(self.training_iterations):
            dataset = []
            print(f"[train]: starting dataset creation at iteration n {i}")

            # decaying std
            self.std = self.std - 0.025*self.std
            self.run.log({"std": self.std})
            
            # alternating task
            if self.alternating_step > 0:
                if (i % self.alternating_step) == 0:
                    env = self.switch_env()
                    self.std = 1
            else:
                env = self.env1

            with torch.no_grad():
                
                self.eval_model(env)
                adv_list = []
                cum_reward = 0
                for _ in range(N): #for each actor

                    # initialize first state
                    s_prime, _ = env.reset()
                    s_prime = torch.tensor(s_prime, dtype=torch.float32).to(self.device)
                    
                    trajectory = []

                    for t in range(T):
                        
                        actions_list_return = [self.actor_list[i](s_prime) for i in range(self.n_nets)] # [(action, mu, std), ..., ]
                        log_policy_list = [self.actor_list[i].get_log_probability(actions_list_return[i][0],
                                                                                actions_list_return[i][1],
                                                                                actions_list_return[i][2]) for i in range(self.n_nets)] # [log_policy, ..., ]
                        action_zero = actions_list_return[0][0]

                        s, reward, terminated, _, _ = env.step(action_zero.squeeze(0).cpu().detach().numpy())
                        s = torch.tensor(s, dtype=torch.float32).to(self.device)
                        reward = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
                        s_prime = s_prime.unsqueeze(0)
                        trajectory.append([s_prime, action_zero, reward, log_policy_list])
                        s_prime = s
                        cum_reward += reward

                        if terminated:
                            break
                        

                    dynamic_target = 0 if terminated else self.critic(s)
                    for t in range(len(trajectory)-1, -1, -1): #I want the range from [T-1 to 0]
                        
                        dynamic_target = trajectory[t][2] + self.gamma*dynamic_target #trajectory[t][2] = reward
                        advantage = dynamic_target - self.critic(trajectory[t][0])
                        trajectory[t] = tuple(trajectory[t] + [dynamic_target.unsqueeze(0), advantage.unsqueeze(0)])

                        dataset.append(trajectory[t])
                        adv_list.append(advantage)

                adv_std, adv_mean = torch.std_mean(torch.tensor(adv_list))
                print(f"[training]: avg cum reward {cum_reward / N}")
                if self.toggle_log:
                    self.run.log({
                        "avg cum_reward": cum_reward / N,
                        "adv mean": adv_mean,
                        "adv std": adv_std,
                        })

            print(f"[training]: ending dataset creation with dataset size {len(dataset)}")

            # Starts the training process
            for e in range(self.epochs):
                
                #print(f"[train]: epoch n {e}")
                avg_loss_value = 0
                avg_loss_ppo = 0
                rd.shuffle(dataset) #shuffle in-place
                
                assert(self.batch_size <= len(dataset))

                for mini_idx in range(0, len(dataset), self.batch_size):
                    
                    # form mini_batch
                    mini_batch = dataset[mini_idx: mini_idx+self.batch_size]
                    # s, action, reward, log, target, advantage
                    state_mini = torch.stack(list(map(lambda elem: elem[0].squeeze(), mini_batch)))
                    action_mini = torch.stack(list(map(lambda elem: elem[1].squeeze(), mini_batch)))
                    total_log_policy_list = torch.t(torch.stack(list(map(lambda elem: torch.cat(elem[3]), mini_batch)))) # size (nets, batch)
                    target_mini = torch.stack(list(map(lambda elem: elem[4].squeeze(), mini_batch)))
                    advantage_mini = torch.stack(list(map(lambda elem: elem[5].squeeze(), mini_batch)))
                    
                    # Normalize advantage_mini
                    advantage_mini = ((advantage_mini-adv_mean) / (adv_std + 1e-8))
                    new_actions = [actor(state_mini) for actor in self.actor_list]

                    total_new_log_policy_list = torch.stack([self.actor_list[i].get_log_probability(action_mini, new_actions[i][1], new_actions[i][2]) for i in range(self.n_nets)]) # size (nets, batch)

                    new_value_mini = self.critic(state_mini)
                    
                    [actor.zero_grad() for actor in self.actor_list]
                    self.critic.optimizer.zero_grad()
                    
                    # returns a loss for every net
                    loss_ppo = self.loss_kl_pc(total_new_log_policy_list, total_log_policy_list, advantage_mini)
                    loss_value = self.loss_value(new_value_mini, target_mini)

                    avg_loss_ppo += loss_ppo
                    avg_loss_value += loss_value

                    loss_ppo.backward()
                    loss_value.backward()

                    # gradient clipping
                    [torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.8) for actor in self.actor_list]
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.8)
                    
                    # optimization step
                    [actor.optimizer.step() for actor in self.actor_list]
                    self.critic.optimizer.step()

                total_minibatch = math.floor(len(dataset) // self.batch_size)
                print(f"epoch {e} -> [avg actor loss]: {avg_loss_ppo / total_minibatch}  [critic loss]: {avg_loss_value / total_minibatch}")
                if self.toggle_log:
                    self.run.log({
                            "avg_loss_ppo": avg_loss_ppo / total_minibatch,
                            "avg_loss_value": avg_loss_value / total_minibatch,
                        })

            #self.save_parameters("partial_models/model"+str(i)+".pt")

    def loss_value(self, value, target):
        #MSE
        return torch.mean((value-target)**2)

    def loss_clip(self, new_log_policy_mini, log_policy_mini, advantage_mini):

        prob_mini = torch.exp(new_log_policy_mini - log_policy_mini)
        prob_adv = prob_mini*advantage_mini
        clip_ = torch.clip(prob_mini, 1-self.epsilon, 1+self.epsilon)*advantage_mini
        return -torch.min(prob_adv, clip_).mean()
    
    def loss_kl_standard(self, new_log_policy_mini, log_policy_mini, advantage_mini):
        
        prob_mini = torch.exp(new_log_policy_mini - log_policy_mini)
        prob_adv = prob_mini * advantage_mini
        d = new_log_policy_mini - log_policy_mini

        if self.mode == 'kl_adaptive':
            if d.detach().mean() < (self.d_targ / 1.5):
                self.beta = self.beta / 2.3
            elif d.detach().mean() > (self.d_targ * 1.5):
                self.beta = self.beta * 1.9

        return -(prob_adv - self.beta*d).mean()
    
    def loss_kl_pc(self, stack_new, stack_old, advantage_mini):
        
        # stack new and stack old have shape (n_net, batch_size)
        # We compute the policy gradient based on first net prob and adv
        new_log_policy_mini = stack_new[0, :]
        log_policy_mini = stack_old[0, :]
        prob_mini = torch.exp(new_log_policy_mini - log_policy_mini)
        L_pg = prob_mini * advantage_mini

        kl_stack = stack_new - stack_old
        L_ppo = torch.sum(self.loss_coeff * torch.t(kl_stack), dim=1)

        L_casc_init = self.omega12 * (torch.t(stack_new)[:, 0] - torch.t(stack_old)[:, 1])
        kl_sub_previous = torch.t(stack_new)[:, 1:] - torch.t(stack_old)[:, 0:stack_old.shape[0]-1]
        # I'm appending to the matrix a row which is equal to the last row. At the end i will have a matrix with
        # We also don't need the first two columns of old
        if self.n_nets > 2:
            kl_sub_successive_second = torch.t(torch.cat((stack_old, stack_new[stack_new.shape[0]-1, :].unsqueeze(0)), 0))[:, 2:]
        else:
            kl_sub_successive_second = 0
        kl_sub_successive = torch.t(stack_new)[:, 1:] - kl_sub_successive_second
        L_casc = L_casc_init + torch.sum(self.omega*kl_sub_previous + kl_sub_successive, dim=1) # summing on net dimension
        
        if self.toggle_log:
            self.run.log({
                "L_pg": L_pg.mean(),
                "L_ppo": L_ppo.mean(),
                "L_casc": L_casc.mean()
            })

        return -(L_pg - L_ppo - L_casc).mean()
    

    def extract_states_prime(self, trajectory):
        return list(map(lambda x: x[0], trajectory))
    
    def save_parameters(self, path):
        torch.save({
            "model_state_dict": self.state_dict()
        }, path + "/model_state_dict.pt")

        torch.save({
            "beta": self.beta,
            "omega": self.omega,
            "omega12": self.omega12,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "std": self.std,
            "d_targ": self.d_targ,
            "n_nets": self.n_nets,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }, path + "/config.pt")

    def load_parameters(self, path):

        # loading parameters
        checkpoint = torch.load(path + "/model_state_dict.pt", map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])

        # loading configuration
        config = torch.load(path + "/config.pt")
        self.beta = config["beta"]
        self.omega = config["omega"]
        self.omega12 = config["omega12"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.std = config["std"]
        self.d_targ = config["d_targ"]
        self.n_nets = config["n_nets"]
        self.actor_learning_rate = config["actor_learning_rate"]
        self.critic_learning_rate = config["critic_learning_rate"]

    def eval_model(self, env):

        total_run = 5
        cum_reward = 0
        for _ in range(total_run):
            s_prime, _ = env.reset()
            s_prime = torch.tensor(s_prime, dtype=torch.float32).to(self.device)
            for _ in range(self.trajectory_length):

                if self.mode == 'pc':
                    _, mu, _ = self.actor_list[0](s_prime)
                else:
                    _, mu, _ = self.actor(s_prime)

                s, reward, terminated, _, _ = env.step(mu.squeeze(0).cpu().detach().numpy())
                s = torch.tensor(s, dtype=torch.float32).to(self.device)
                s_prime = s_prime.unsqueeze(0)
                s_prime = s
                cum_reward += reward
                if terminated:
                    break

        print("avg cum_reward_eval: ", cum_reward / total_run)
        if (cum_reward / total_run) > 800:
            self.save_parameters(path="partial_models"+str(rd.randint(0, 1000)))
            
        if self.toggle_log:
            self.run.log({
                "avg cum_reward_eval": cum_reward / total_run
            })

if __name__ == '__main__':
    
    env_space = "HalfCheetah-v5"
    env1 = gym.make(env_space, ctrl_cost_weight=0.1)
    env2 = HalfCheetahEnv(xml_file="half_cheetah_bigleg.xml", ctrl_cost_weight=0.1)

    epochs = 10
    training_iterations = 20
    batch_size = 64
    trajectory_length = 1000
    n_actors = 10
    in_features = env1.observation_space.shape[0]
    out_features = env1.action_space.shape[0]
    hidden_features = 64
    actor_learning_rate = 5e-4
    critic_learning_rate = 5e-3
    gamma = 0.99
    lambda_ = 0.95
    epsilon = 0.2
    beta = 0.5
    omega = 4
    omega12 = 1
    d_targ = 1
    std = 0.25
    n_nets = 1
    toggle_log = False
    alternating_step = 0

    device = "mps"
    mode = "clip"

    if mode == "pc":
        assert(n_nets > 1)
    modes = ["kl_fixed", "kl_adaptive", "clip", "pc"]
    assert(mode in modes)

    if toggle_log:
        run = wandb.init(
            entity="alecoccia-sapienza-universit-di-roma",
            project="RL",
            config = {
                "env_name": env_space,
                "epochs": epochs,
                "training_iterations": training_iterations,
                "batch_size": batch_size,
                "trajectory_length": trajectory_length,
                "n_actors": n_actors,
                "in_features": in_features,
                "out_features": out_features,
                "hidden_features": hidden_features,
                "actor_learning_rate": actor_learning_rate,
                "critic_learning_rate": critic_learning_rate,
                "gamma": gamma,
                "lambda_": lambda_,
                "epsilon": epsilon,
                "beta": beta,
                "omega": omega,
                "omega12": omega12,
                "d_targ": d_targ,
                "std": std,
                "n_nets": n_nets,
                "device": device,
                "mode": mode,
                "alternating_step": alternating_step,
            },
        )
    else:
        run = None

    device = 'cpu'

    ppo = PPO(epochs=epochs,
            training_iterations=training_iterations,
            batch_size=batch_size,
            trajectory_length=trajectory_length, 
            n_actors=n_actors,
            env1=env1,
            env2=env2,
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            device=device,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            gamma=gamma,
            lambda_=lambda_,
            epsilon=epsilon,
            beta = beta,
            d_targ=d_targ,
            std=std,
            mode=mode,
            n_nets=n_nets,
            omega=omega,
            omega12=omega12,
            run=run,
            toggle_log=toggle_log,
            alternating_step=alternating_step,
    )

    ppo.load_state_dict(torch.load("final_models/clip_single/model_state_dict.pt"))

    if toggle_log:
        name = "clip_single2"
        #ppo.load_parameters("final_models/" + name)

        ppo.train_manager()
        
        os.makedirs("final_models/" + name, exist_ok=True)
        ppo.save_parameters("final_models/" + name)

        run.finish()