import torch
import dataloader as D
import config as C

import random

print(C.seen_classes)
print(C.unseen_classes)

x = random.sample(C.seen_classes, k=3)

# print(D.ISCXVPN2016.paths())

print(D.train_seen["icq"])
