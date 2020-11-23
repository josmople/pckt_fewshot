import fewshot as F
import torch

import data as D
import config as C

d = D.ISCXVPN2016(C.dirpath_iscxvpn2016, allow_pcapng=False)

for t in d.tags():
    print(t)
