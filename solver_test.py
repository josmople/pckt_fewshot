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

network = M.SimpleClassifier(in_channels=C.usable_bits, hidden_channels=[C.bits, C.bits, C.bits // 4, C.bits, C.usable_bits], n_classes=C.usable_bits).cuda()

classes = random.sample(C.seen_classes, k=C.episode_size)
datasets = [D.train_seen[c] for c in classes]

loss = F.accuracy(*datasets, features_fn=network, n_support=C.shots)
