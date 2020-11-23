import torch
import data as D
import config as C

import torch

from tqdm import tqdm


ISCXVPN2016 = D.ISCXVPN2016(C.dirpath_iscxvpn2016, allow_pcapng=False)


def generate_pcap_dataset(path, size=40, verbose=False, lazy=False):
    from torch import tensor
    from pcapfile.linklayer import ethernet

    def transform(packet):
        raw_bytes = ethernet.strip_ethernet(packet.raw())
        raw_size = len(raw_bytes)
        missing_size = max(0, size - raw_size)
        missing_bytes = bytes([0] * missing_size)
        all_data = raw_bytes + missing_bytes
        data = all_data[:size]
        data = data[:12] + data[20:]
        bitstring = str.join("", map(lambda n: f"{n:08b}", data))
        bitarray = [float(c) for c in bitstring]
        return tensor(bitarray)

    ds = D.PCAPDataset(path=path, transform=transform, layers=0, lazy=lazy, verbose=verbose)
    return ds


def load_pcap_dataset_from_iscxvpn2016_by_name(name):
    pass


def load_pcap_dataset_from_iscxvpn2016(pos_cls, neg_cls=None, transform=None):

    idxs = ISCXVPN2016.find(pos_cls, neg_cls)
    all_paths = ISCXVPN2016.paths()

    datasets = []
    paths = []

    faulty = []

    for idx in idxs:
        path = all_paths[idx]
        paths.append(path)

        try:
            dataset = generate_pcap_dataset(path, size=C.bytecount)
            datasets.append(dataset)
        except:
            faulty.append(idx)

    for f_idx in faulty:
        idxs.remove(f_idx)

    if len(faulty) > 0:
        print(f"For pos={pos_cls} and  neg={neg_cls}, cannot read {[all_paths[i] for i in faulty]}")

    if len(datasets) == 0:
        raise Exception(f"No dataset found: Pos={pos_cls}, Neg={neg_cls}")

    if len(datasets) > 1:
        datasets = sum(datasets[1:], datasets[0])
    else:
        datasets = datasets[0]

    meta = {
        "idxs": idxs,
        "paths": paths,
        "datasets": datasets
    }

    return D.utils.dmap(datasets, transform), meta


def load_dataset(cls):
    pos, *neg = cls.split("-")
    neg = neg[0] if len(neg) > 0 else ""

    pos_cls = pos.split("_")
    neg_cls = neg.split("_")
    print(f"Loading: Pos={pos_cls}, Neg={neg_cls}")

    ds = load_pcap_dataset_from_iscxvpn2016(pos_cls, neg_cls, lambda t: t.cuda())[0]
    return ds


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


train_seen = {}
test_seen = {}
test_unseen = {}


def init():
    print("Seen Classes: ", C.seen_classes)
    print("UnSeen Classes: ", C.unseen_classes)
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
