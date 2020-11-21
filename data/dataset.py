from . import utils as _D


class PCAPDataset(_D.dataset.ValueDataset):

    def __init__(self, path, transform, layers=0, verbose=False, lazy=False):
        from pcapfile.savefile import load_savefile

        self.file_raw = open(path, 'rb')
        self.file_parsed = load_savefile(self.file_raw, layers=layers, verbose=verbose, lazy=lazy)

        super().__init__(self.file_parsed.packets, transform)

    def close(self):
        self.file_raw.close()


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

    ds = PCAPDataset(path=path, transform=transform, layers=0, lazy=lazy, verbose=verbose)
    return ds
