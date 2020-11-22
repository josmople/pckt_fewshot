
all_classes = ["aim", "icq", "email", "facebook", "netflix"]
# all_classes = ["aim", "icq", "torrent", "email", "facebook", "ftps", "hangouts", "netflix", "scp", "sftp", "skype", "vimeo", "voipbuster", "youtube"]
# all_classes = ["aim", "icq", "email", "facebook", "netflix", "scp", "sftp", "skype", "vimeo", "voipbuster", "youtube"]
seen_classes = all_classes[:len(all_classes) // 4 * 3]
unseen_classes = [c for c in all_classes if c not in seen_classes]

episode_count = 100000
episode_size = max(len(seen_classes) // 4 * 3, 2)
shots = 5

bytecount = 40

lr_init = 1e-3
lr_epoch_updates = [int(10**i) for i in range(1, 5)]

seed = 42

seen_classes_split = 0.8

dirpath_iscxvpn2016 = "D://Datasets/ISCXVPN2016/"


# Constant
ADDRESS_BYTES = 8

# Derived
bits = bytecount * 8
address_bits = ADDRESS_BYTES * 8
usable_bits = bits - address_bits
