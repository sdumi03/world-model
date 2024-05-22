import torch


class MODEL(torch.nn.Module):
    def __init__(self, latent, recurrent, action):
        super().__init__()
        self.fc = torch.nn.Linear(latent + recurrent, action)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)