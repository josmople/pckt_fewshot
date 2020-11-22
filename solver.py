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

optimizer = optim.Adam(network.parameters(), lr=C.lr_init)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.lr_epoch_updates, gamma=0.1)


def step_fn(loss):
    print(globals()['e'], "\t", loss.item(), "\t", globals()['classes'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


good = True
counter = 0
prev_loss = 10000
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

    if e % 500 == 0:
        with torch.no_grad():
            classes = random.sample(C.unseen_classes, k=C.episode_size)
            datasets = [D.test_unseen[c] for c in classes]
            acc = F.accuracy(*datasets, features_fn=network, n_support=C.shots)
            print("Accuracy: ", acc)

    if e % 100 == 0:
        loss = loss.item()
        if loss < prev_loss:
            if good:
                counter += 1
            else:
                counter = 1
            good = True
            print(f"\tGood {counter:02} {e:05}", loss)
        else:
            if not good:
                counter += 1
            else:
                counter = 1
            good = False
            print(f"\tBad {counter:02} {e:05}", loss)
        prev_loss = loss

    scheduler.step()

print("Final: ", loss.item())
torch.save(network.state_dict(), "test.pth")
print("End")
