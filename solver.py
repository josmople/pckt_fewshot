import torch
import torch.optim as optim

from tqdm import tqdm

import dataloader as D
import model as M
import config as C
import fewshot as F

import random
random.seed(C.seed)
torch.manual_seed(C.seed)

network = M.SimpleClassifier(in_channels=C.usable_bits, hidden_channels=[C.bits, C.bits, C.bits, C.bits, C.usable_bits], n_classes=C.usable_bits).cuda()

optimizer = optim.Adam(network.parameters(), lr=C.lr_init)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.lr_epoch_updates, gamma=0.1)


def step_fn(loss):
    print(globals()['e'], "\t", loss.item(), "\t", globals()['classes'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for e in tqdm(range(C.episode_count)):
    classes = random.sample(C.seen_classes, k=C.episode_size)
    # print(classes)
    datasets = [D.train_seen[c] for c in classes]
    # loss = F.episode(*datasets, features_fn=network, step_fn=step_fn, shots=C.shots)

    loss = F.episode(*datasets, features_fn=network, n_support=C.shots)
    # print("\t", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 100 == 0:
        print("\t", loss.item())

    scheduler.step()

print("Final: ", loss.item())
print("End")
