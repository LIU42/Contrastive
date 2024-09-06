import torch
import yaml

from models import ContrastModel


def load_configs():
    with open('configs/model.yaml', 'r') as configs:
        return yaml.safe_load(configs)


configs = load_configs()

model = ContrastModel(pretrained=configs['pretrained'])
model.eval()

torch.save(model.state_dict(), configs['save-path'])
