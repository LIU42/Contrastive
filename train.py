import torch
import yaml

import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from models import ContrastModel
from dataset import CaptionDataset


def criterion(image_features, text_features, temperature):
    predicts = image_features @ text_features.T
    predicts /= temperature

    labels = torch.arange(predicts.size(0))
    labels = labels.cuda()

    i2t_loss = functional.cross_entropy(predicts.permute(0, 1), labels)
    t2i_loss = functional.cross_entropy(predicts.permute(1, 0), labels)

    return i2t_loss + t2i_loss


def validation(image_features, text_features):
    predicts = image_features @ text_features.T

    labels = torch.arange(predicts.size(0))
    labels = labels.cuda()

    correct_i2t = (torch.argmax(predicts.permute(0, 1), dim=1) == labels).sum().item()
    correct_t2i = (torch.argmax(predicts.permute(1, 0), dim=1) == labels).sum().item()

    return correct_i2t, correct_t2i


with open('configs/train.yaml', 'r') as configs:
    configs = yaml.safe_load(configs)

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.5, 2.0)),
])

normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if configs['use-augment']:
    train_transform = augment_transform
else:
    train_transform = normal_transform

valid_transform = normal_transform

train_dataset = CaptionDataset('datasets/train', train_transform)
valid_dataset = CaptionDataset('datasets/valid', valid_transform)

train_loader = data.DataLoader(train_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])
valid_loader = data.DataLoader(valid_dataset, configs['batch-size'], shuffle=True, num_workers=configs['num-workers'])

load_path = configs['load-path']
best_path = configs['best-path']
last_path = configs['last-path']

model = ContrastModel(pretrained=configs['load-pretrained'])
model = model.cuda()

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_path, map_location='cuda', weights_only=True))

scaler = torch.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=configs['learning-rate'])

best_accuracy = 0.0
temperature = configs['temperature']

print(f'\n---------- Training Start ----------\n')

for epoch in range(configs['epochs']):
    model.train()
    training_loss = 0.0

    for index, (images, texts, masks) in enumerate(train_loader, start=1):
        images = images.cuda()
        texts = texts.cuda()
        masks = masks.cuda()
        optimizer.zero_grad()

        if configs['use-amp']:
            with torch.autocast('cuda'):
                loss = criterion(model.encode_image(images), model.encode_text(texts, masks), temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model.encode_image(images), model.encode_text(texts, masks), temperature)
            loss.backward()
            optimizer.step()

        training_loss += loss.item()

        print(f'\rBatch Loss: {loss:.5f} [{index}/{len(train_loader)}]', end='')

    model.eval()
    training_loss /= len(train_loader)

    with torch.no_grad():
        accuracy_i2t = 0
        accuracy_t2i = 0

        for images, texts, masks in valid_loader:
            images = images.cuda()
            texts = texts.cuda()
            masks = masks.cuda()

            if configs['use-amp']:
                with torch.autocast('cuda'):
                    correct_i2t, correct_t2i = validation(model.encode_image(images), model.encode_text(texts, masks))
            else:
                correct_i2t, correct_t2i = validation(model.encode_image(images), model.encode_text(texts, masks))

            accuracy_i2t += correct_i2t
            accuracy_t2i += correct_t2i

    accuracy_i2t /= len(valid_dataset)
    accuracy_t2i /= len(valid_dataset)

    mean_accuracy = 2 * (accuracy_i2t * accuracy_t2i) / (accuracy_i2t + accuracy_t2i)

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), last_path)

    print(f'\tEpoch: {epoch:<6} Loss: {training_loss:<10.5f} I2T: {accuracy_i2t:<6.2f} T2I: {accuracy_t2i:<6.2f}')

print('\n---------- Training Finish ----------\n')
print(f'Best Accuracy: {best_accuracy:.5f}')
