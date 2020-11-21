

class ISCXVPN2016:

    def __init__(self, dirpath, allow_pcap=True, allow_pcapng=True):
        self.dirpath = dirpath
        self.allow_pcap = allow_pcap
        self.allow_pcapng = allow_pcapng
        self.cache = {}

    def clear_cache(self):
        self.cache = {}

    def paths(self):
        if "paths" in self.cache:
            return self.cache["paths"]

        from glob import glob
        files = glob(f"{self.dirpath}/*")

        output = []
        if self.allow_pcap:
            output += list(filter(lambda f: f.endswith(".pcap"), files))
        if self.allow_pcapng:
            output += list(filter(lambda f: f.endswith(".pcapng"), files))

        output = sorted(output)
        self.cache["paths"] = output
        return output

    def names(self):
        if "names" in self.cache:
            return self.cache["names"]

        from os.path import basename, splitext
        output = list(map(lambda f: splitext(basename(f))[0], self.paths()))

        self.cache["names"] = output
        return output

    def filenames(self):
        if "filenames" in self.cache:
            return self.cache["filenames"]

        from os.path import basename
        output = list(map(basename, self.paths()))

        self.cache["filenames"] = output
        return output

    def tags(self):
        if "tags" in self.cache:
            return self.cache["tags"]

        def tokenize(n):
            from re import sub

            REGEX = [
                '([A-Z]{2,})',
            ]
            for R in REGEX:
                n = sub(R, r'\1 ', n)

            REGEX = [
                '_?(chat)',
                '_([a-z]+)',
                '_?([A-Z]+)',
                '_?([A-Z][a-z]+)',
                '_?([0-9][a-z]*)',
            ]

            for R in REGEX:
                n = sub(R, r' \1', n)

            return n.split()

        def lower_case(tags):
            return map(str.lower, tags)

        def filter_tags(tags):
            def filter_fn(tag: str):
                if any([tag.startswith(str(i)) for i in range(9)]):
                    return False
                if tag == "a" or tag == "b":
                    return False
                return True
            return filter(filter_fn, tags)

        def manual_edits(tags):
            if "bittorrent" in tags:
                idx = tags.index("bittorrent")
                tags[idx] = "torrent"
            if "hangout" in tags:
                idx = tags.index("hangout")
                tags[idx] = "hangouts"
            return tags

        def generate_tags(name):
            return manual_edits(list(filter_tags(lower_case(tokenize(name)))))

        output = [generate_tags(name) for name in self.names()]
        self.cache["tags"] = output
        return output
