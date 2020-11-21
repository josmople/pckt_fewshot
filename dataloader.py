import torch
import data as D
import config as C

import torch


ISCXVPN2016 = D.ISCXVPN2016(C.dirpath_iscxvpn2016, allow_pcapng=False)


def load_dataset_from_scratch(cls):
    paths = ISCXVPN2016.paths()
    tags = ISCXVPN2016.tags()

    datasets = []
    for p, t in zip(paths, tags):
        if cls in t:
            ds = D.generate_pcap_dataset(p)
            datasets.append(ds)

    def close_all():
        for d in datasets:
            d.close()

    if len(datasets) > 0:
        return sum(datasets[1:], datasets[0]), close_all
    return datasets[0], close_all


# load_dataset = load_dataset_from_scratch
def load_dataset(cls):
    from os.path import exists
    if exists(f"cache/{cls}"):
        print("Loading Cache: ", cls)
        from torch import load
        return D.utils.files(f"cache/{cls}/*.pt", load)

    from tqdm import tqdm

    ds, close_all = load_dataset_from_scratch(cls)
    ds = D.utils.dcache_tensor(ds, f"cache/{cls}/{{idx:012}}.pt")
    print("Processing: ", cls)
    ds = D.utils.DataLoader(ds, batch_size=1000, num_workers=1000)
    for _ in tqdm(ds):
        pass
    close_all()

    return load_dataset(cls)


train_seen = {}
test_seen = {}
test_unseen = {}

for cls in C.seen_classes:
    ds = load_dataset(cls)

    train_count = int(C.seen_classes_split * len(ds))
    test_count = len(ds) - train_count

    train_ds, test_ds = D.utils.random_split(ds, [train_count, test_count])

    train_seen[cls] = train_ds
    test_seen[cls] = test_ds

for cls in C.unseen_classes:
    ds = load_dataset(cls)
    test_unseen[cls] = ds
