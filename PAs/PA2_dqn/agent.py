import os
import torch
import platform
import numpy as np
import torch.optim as optim
from copy import deepcopy
from model import QNetwork, DuelingQNetwork

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device='cuda', compile=True):
        self.device = device
        self.use_double = cfg.use_double
        self.use_dueling = cfg.use_dueling
        self.target_update_interval = cfg.target_update_interval
        q_model = DuelingQNetwork if self.use_dueling else QNetwork

        self.q_net = q_model(state_size, action_size, cfg.hidden_size, cfg.activation).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr)

        self.tau = cfg.tau
        self.gamma = cfg.gamma ** cfg.nstep
        if platform.system() == "Linux" and compile:
            # torch.compile is not supported on Windows or MacOS
            self.compile()

    def compile(self):
        self.q_net = torch.compile(self.q_net)
        self.target_net = torch.compile(self.target_net)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def get_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ############################
        # YOUR IMPLEMENTATION HERE #
        q_values = self.q_net(state)
        Q = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        ############################
        return Q
    
    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get the optimal action according to the current Q value and state
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.q_net(state_tensor)
        
        if q_values.dim() == 1: 
            actions = torch.argmax(q_values).cpu().numpy()
        else:  
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        ############################
        return actions

    @torch.no_grad()
    def get_Q_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Get the target Q value according to the Bellman equation
        """
        if self.use_double:
            ##########################
            # YOUR IMPLEMENTATION HERE
            next_q_values = self.q_net(next_state)
            next_actions = torch.argmax(next_q_values, dim=1)
            next_q_values_target = self.target_net(next_state)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            Q_target = reward + (1 - done) * self.gamma * next_q_value
            ##########################
        else:
            ##########################
            # YOUR IMPLEMENTATION HERE
            next_q_values = self.target_net(next_state)
            next_q_value = torch.max(next_q_values, dim=1)[0]
            Q_target = reward + self.gamma * next_q_value * (1 - done)
            ##########################
        return Q_target

    def update(self, batch, step, weights=None):
        state, action, reward, next_state, done = batch

        Q_target = self.get_Q_target(reward, done, next_state)

        Q = self.get_Q(state, action)
        if weights is None:
            weights = torch.ones_like(Q).to(self.device)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not step % self.target_update_interval:
            self.soft_update(self.target_net, self.q_net)

        return loss.item(), td_error, Q.detach().mean().item()

    def save(self, name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join('models', name))

    def load(self, root_path='', name='best_model.pt'):
        self.q_net.load_state_dict(torch.load(os.path.join(root_path, 'models', name)))

    def __repr__(self) -> str:
        use_double = 'Double' if self.use_double else ''
        use_dueling = 'Dueling' if self.use_dueling else ''
        prefix = 'Normal' if not self.use_double and not self.use_dueling else ''
        return use_double + use_dueling + prefix + 'QNetwork'
