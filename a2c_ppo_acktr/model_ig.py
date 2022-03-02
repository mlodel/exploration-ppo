import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.common import imagenet_rgb_preprocess, states_preprocess

from a2c_ppo_acktr.model import NNBase, Flatten

from a2c_ppo_acktr.cnn_encoders import CNN3Layer, ResNetEnc, CNN3Layer_old


class Policy_IG(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy_IG, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            base = IG_Base

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class IG_Base(NNBase):
    def __init__(self, num_inputs, recurrent=False, device='cpu', fix_cnn=False, hidden_size=256):
        super(IG_Base, self).__init__(recurrent, hidden_size, hidden_size)

        self.device = device

        self.num_states = 6
        self.img_size = 80

        self.large_map_encoder = CNN3Layer_old(img_size=self.img_size, img_ch=1, out_ftrs=128)
        self.small_map_encoder = CNN3Layer_old(img_size=self.img_size, img_ch=1, out_ftrs=128)
        cnn_models = [self.large_map_encoder,
                      self.small_map_encoder]
        if fix_cnn:
            for model in cnn_models:
                for name, param in model.named_parameters():
                    if name not in ['network.fc.weight', 'network.fc.bias']:
                        param.requires_grad = False
                    else:
                        continue

        init_linear = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                     constant_(x, 0))
        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))

        self.state_encoder = nn.Sequential(
            init_linear(nn.Linear(self.num_states, 128)),
            # nn.BatchNorm1d(64),
            # nn.ELU(),
            # nn.Linear(64, 128),
        )
        num_in = 3 * 128

        self.merge_fc = nn.Sequential(
            init_relu(nn.Linear(num_in, hidden_size)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            init_relu(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_linear(nn.Linear(hidden_size, 1))
        )

        self.actor = nn.Sequential(
            init_relu(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.train()

    def forward(self, obs, rnn_hxs, masks):

        states_obs = obs[:, 0:self.num_states]
        small_map_obs = obs[:, self.num_states:-self.img_size ** 2].reshape((-1, self.img_size, self.img_size))
        large_map_obs = obs[:, -self.img_size ** 2:].reshape((-1, self.img_size, self.img_size))

        large_maps = imagenet_rgb_preprocess(large_map_obs, device=self.device)
        small_maps = imagenet_rgb_preprocess(small_map_obs, device=self.device)
        states, obs_states = states_preprocess(states_obs, device=self.device)

        l_cnn_out = self.large_map_encoder(large_maps)
        s_cnn_out = self.small_map_encoder(small_maps)
        st_fc_out = self.state_encoder(states)

        cnn_out = torch.cat((l_cnn_out, s_cnn_out, st_fc_out), dim=-1)
        latent = self.merge_fc(cnn_out)

        if self.is_recurrent:
            latent, rnn_hxs = self._forward_gru(latent, rnn_hxs, masks)

        return self.critic(latent), self.actor(latent), rnn_hxs
