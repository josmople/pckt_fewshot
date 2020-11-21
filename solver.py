import torch
import torch.optim as optim

import dataloader as D
import model as M
import config as C
import fewshot as F

import random
random.seed(C.seed)
torch.manual_seed(C.seed)

network = M.SimpleClassifier()

optimizer = optim.Adam(network.parameters(), lr=C.lr_init)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.lr_epoch_updates, gamma=0.1)

for e in range(C.episode_count):
    classes = random.sample(C.seen_classes, k=C.episode_size)
    loss = F.episode(*[D.train_seen[c] for c in classes], network, shots=C.shots)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
