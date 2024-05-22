ACTION_SIZE = 2
LATENT_SIZE = 128 # 32
RECURRENT_SIZE = 256
RED_SIZE = 64
SIZE = 64
IMAGE_CHANNELS = 30 # 3
SEQ_LEN = 4 # 1  # 32


import torch
import numpy as np
from models import vae, mdrnn, controller
from envs import trading
from os.path import join, exists


def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()


def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened


def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


class RolloutGenerator:
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, trained_dir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        vae_file, rnn_file, ctrl_file = [
            join(trained_dir, model, 'best.tar')
            for model in ['vae', 'mdrnn', 'controller']
        ]

        assert exists(vae_file) and exists(rnn_file), 'Either vae or mdrnn is untrained'

        vae_state, rnn_state = [
            torch.load(file_name, map_location={'cuda:0': str(device)})
            for file_name in (vae_file, rnn_file)
        ]

        # for model, state in (('VAE', vae_state), ('MDRNN', rnn_state)):
        #     print(f"Loading {model} at epoch {state['epoch']} with test loss {state['precision']}")

        self.vae = vae.MODEL(IMAGE_CHANNELS, LATENT_SIZE, '1d').to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = mdrnn.MDRNNCell(LATENT_SIZE, ACTION_SIZE, RECURRENT_SIZE, 5).to(device)
        self.mdrnn.load_state_dict({
            k.strip('_l0'): v
            for k, v in rnn_state['state_dict'].items()
        })

        self.controller = controller.MODEL(LATENT_SIZE, RECURRENT_SIZE, ACTION_SIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print(f"Loading Controller with reward {ctrl_state['reward']}")
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = trading.Env()
        self.device = device
        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        # print('obs', obs.shape)
        # import numpy as np
        # print('hidden', np.array(hidden).shape)
        obs = torch.tensor(obs).float().unsqueeze(0)

        _, latent_mu, _ = self.vae(obs)
        action = self.controller(latent_mu, hidden[0])
        action = torch.argmax(action).unsqueeze(0).unsqueeze(0)
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)

        # print('action', action.squeeze().shape)
        # print('next_hidden', np.array(next_hidden).shape)

        return np.argmax(action.squeeze().cpu().numpy()), next_hidden

    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        # self.env.render()

        hidden = [
            torch.zeros(1, RECURRENT_SIZE).to(self.device)
            for _ in range(2)
        ]

        cumulative = 0
        i = 0
        while True:
            # Este transform solo en '2d'
            # from torchvision import transforms
            # obs = transform(obs).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs, hidden)
            obs, reward, done, _, _ = self.env.step(action)

            # if render: self.env.render()

            cumulative += reward

            if done or i > self.time_limit: return - cumulative

            i += 1
