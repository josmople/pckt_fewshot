
# all_classes = ["aim", "icq", "torrent", "email"]
# all_classes = ["aim", "icq", "torrent", "email", "facebook", "ftps", "hangouts", "netflix", "scp", "sftp", "skype", "vimeo", "voipbuster", "youtube"]
all_classes = ["aim", "icq", "email", "facebook", "netflix", "scp", "sftp", "skype", "vimeo", "voipbuster", "youtube"]
seen_classes = all_classes[:len(all_classes) // 4 * 3]
unseen_classes = [c for c in all_classes if c not in seen_classes]

episode_count = 1000
episode_size = 5

seen_classes_split = 0.8

dirpath_iscxvpn2016 = "D://Datasets/ISCXVPN2016/"
