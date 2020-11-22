import torch
import data as D
import config as C

import torch

from tqdm import tqdm


ISCXVPN2016 = D.ISCXVPN2016(C.dirpath_iscxvpn2016, allow_pcapng=False)


def load_dataset_from_scratch(cls):
    paths = ISCXVPN2016.paths()
    tags = ISCXVPN2016.tags()

    pos, *neg = cls.split("-")
    neg = neg[0] if len(neg) > 0 else ""

    pos_cls = pos.split("_")
    neg_cls = neg.split("_")

    print(f"Loading: Pos={pos_cls}, Neg={neg_cls}")

    datasets = []
    for p, t in zip(paths, tags):
        if all([c in t for c in pos_cls]) and all([c not in t for c in neg_cls]):
            ds = D.generate_pcap_dataset(p)
            datasets.append(ds)

    def close_all():
        for d in datasets:
            d.close()

    if len(datasets) > 1:
        return sum(datasets[1:], datasets[0]), close_all
    if len(datasets) > 0:
        return datasets[0], close_all
    raise Exception(f"No dataset found: Pos={pos_cls}, Neg={neg_cls}")


def load_dataset(cls):
    ds = load_dataset_from_scratch(cls)[0]
    return D.utils.dmap(ds, lambda t: t.cuda())


# def load_dataset(cls):
#     from os.path import exists
#     if exists(f"cache/{cls}"):
#         print("Loading Cache: ", cls)
#         from os import walk
#         from torch import load

#         files = next(walk(f"cache/{cls}/"))[2]
#         N = len(files)

#         print(f"For class {cls} count is {N}")
#         return D.utils.dmap(range(N), [
#             lambda idx: f"cache/{cls}/{idx:012}.pt",
#             load,
#             lambda t: t.cuda()]
#         )

#     from tqdm import tqdm

#     ds, close_all = load_dataset_from_scratch(cls)
#     ds = D.utils.dcache_tensor(ds, f"cache/{cls}/{{idx:012}}.pt")
#     print("Processing: ", cls)
#     for _ in tqdm(ds):
#         pass
#     close_all()

#     return load_dataset(cls)


print("Seen Classes: ", C.seen_classes)
print("UnSeen Classes: ", C.unseen_classes)

train_seen = {}
test_seen = {}
test_unseen = {}

print(f"Loading seen classes")

for cls in tqdm(C.seen_classes):
    print(f"Dataset {cls} is loading")
    ds = load_dataset(cls)
    # print(f"Dataset {cls} is loaded")

    train_count = int(C.seen_classes_split * len(ds))
    test_count = len(ds) - train_count

    print(f"Splitting dataset {cls} at [{train_count}, {test_count}]")
    train_ds, test_ds = D.utils.random_split(ds, [train_count, test_count])
    # print(f"Split done")

    train_seen[cls] = train_ds
    test_seen[cls] = test_ds

print(f"Loading unseen classes")

for cls in tqdm(C.unseen_classes):
    ds = load_dataset(cls)
    print(f"Dataset {cls} is loaded")
    test_unseen[cls] = ds
