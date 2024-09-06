import torch
import yaml

import torch.utils.data as data
import torchvision.transforms as transforms

from models import ContrastModel
from dataset import CaptionDataset


def load_configs():
    with open('configs/eval.yaml', 'r') as configs:
        return yaml.safe_load(configs)


configs = load_configs()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CaptionDataset('datasets/valid', transform)
dataset_size = len(dataset)

dataloader = data.DataLoader(dataset, configs['batch-size'], num_workers=configs['num-workers'])
dataloader_size = len(dataloader)

model = ContrastModel(pretrained=False)
model = model.cuda()
model.load_state_dict(torch.load(configs['model-path'], map_location='cuda', weights_only=True))

image_features = []
text_features = []

i2t_recall01 = 0
i2t_recall05 = 0
i2t_recall10 = 0

t2i_recall01 = 0
t2i_recall05 = 0
t2i_recall10 = 0

print(f'\n---------- Evaluation Start ----------\n')

with torch.no_grad():
    model.eval()

    for index, (images, texts, masks) in enumerate(dataloader, start=1):
        images = images.cuda()
        texts = texts.cuda()
        masks = masks.cuda()

        if configs['use-amp']:
            with torch.autocast('cuda'):
                image_feature, text_feature = model(images, texts, masks)
        else:
            image_feature, text_feature = model(images, texts, masks)

        image_features.append(image_feature)
        text_features.append(text_feature)

        print(f'\rEncoding: [{index}/{dataloader_size}]', end=' ')

    print('\nEncoding Done!\n')

    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)

    for index in range(dataset_size):
        i2t_results = image_features[index] @ text_features.T

        _, top01_indices = torch.topk(i2t_results, 1)
        _, top05_indices = torch.topk(i2t_results, 5)
        _, top10_indices = torch.topk(i2t_results, 10)

        if index in top01_indices:
            i2t_recall01 += 1
        if index in top05_indices:
            i2t_recall05 += 1
        if index in top10_indices:
            i2t_recall10 += 1

        t2i_results = text_features[index] @ image_features.T

        _, top01_indices = torch.topk(t2i_results, 1)
        _, top05_indices = torch.topk(t2i_results, 5)
        _, top10_indices = torch.topk(t2i_results, 10)

        if index in top01_indices:
            t2i_recall01 += 1
        if index in top05_indices:
            t2i_recall05 += 1
        if index in top10_indices:
            t2i_recall10 += 1

        print(f'\rEvaluating: [{index}/{dataset_size}]', end=' ')

    print('\nEvaluating Done!\n')

i2t_recall01 = i2t_recall01 / dataset_size
i2t_recall05 = i2t_recall05 / dataset_size
i2t_recall10 = i2t_recall10 / dataset_size

t2i_recall01 = t2i_recall01 / dataset_size
t2i_recall05 = t2i_recall05 / dataset_size
t2i_recall10 = t2i_recall10 / dataset_size

print(f'I2T R01: {i2t_recall01:.5f}')
print(f'I2T R05: {i2t_recall05:.5f}')
print(f'I2T R10: {i2t_recall10:.5f}')

print('\n------------------\n')

print(f'T2I R01: {t2i_recall01:.5f}')
print(f'T2I R05: {t2i_recall05:.5f}')
print(f'T2I R10: {t2i_recall10:.5f}')

print(f'\n---------- Evaluation End ----------\n')
