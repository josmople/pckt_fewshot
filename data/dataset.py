from . import utils as _D


class PCAPDataset(_D.dataset.ValueDataset):

    def __init__(self, path, transform, layers=0, verbose=False, lazy=False):
        from pcapfile.savefile import load_savefile

        self.file_raw = open(path, 'rb')
        self.file_parsed = load_savefile(self.file_raw, layers=layers, verbose=verbose, lazy=lazy)

        super().__init__(self.file_parsed.packets, transform)

    def close(self):
        self.file_raw.close()
