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


network = M.SimpleClassifier(in_channels=C.usable_bits, hidden_channels=[C.bits * 8, C.bits * 4, C.bits * 2, C.bits, C.usable_bits], n_classes=C.usable_bits).cuda()
network.load_state_dict(torch.load("weights/2020-11-23.pth"))

seen_classes = ["aim", "icq", "facebook_chat-vpn", "facebook_audio-vpn", "hangouts_chat-vpn", "vimeo", "voipbuster"]

test_classes = {
    # "Seen": ["aim", "icq", "vimeo", "voipbuster"],
    # "Seen/Unseen": ["aim", "vimeo", "hangouts_audio", "twitter"],
    # "Generic": ["audio", "chat", "file", "email"],
    # "Application": ["facebook_audio-video", "spotify", "hangouts_audio", "skype-video", "google", "twitter"],
    "Connection": ["vpn", "tor"]
}
for name, classes in test_classes.items():
    datasets = [D.load_dataset(c) for c in classes]
    print("____________________________________________________________________")
    print("NAME: ", name)
    print("CLASSES: ", classes)
    print("SHOTS, ACCURACY:")
    with torch.no_grad():
        for shot in [1, 5, 10, 20, 100]:
            acc = F.accuracy(*datasets, features_fn=network, n_support=shot, n_query=5000)
            print(shot, ", ", acc)
