
def select_batch(dataset, size, **kwds):
    from torch.utils.data import DataLoader
    for x in DataLoader(dataset, batch_size=size, shuffle=True, **kwds):
        return x


def prototype_fn(x):
    from torch import mean
    return mean(x, dim=0, keepdims=True)


def distance_fn(x, y):
    from torch import dist
    return dist(x, y, p=2)


def episode(*datasets, features_fn, prototype_fn=prototype_fn, distance_fn=distance_fn, shots=5):
    loss = 0
    for dataset in datasets:
        batch = select_batch(dataset, shots + 1)

        query = batch[shots:]
        query = features_fn(query)

        support = batch[0:shots]
        support = features_fn(support)

        prototype = prototype_fn(support)
        loss += distance_fn(prototype, query)

    loss /= len(datasets)
    return loss


def predict(supports, queries, features_fn, prototype_fn=prototype_fn, distance_fn=distance_fn):
    distances = []
    for support, query in zip(supports, queries):
        query = features_fn(query)
        support = features_fn(support)

        prototype = prototype_fn(support)
        distance = distance_fn(prototype, query)
        distances.append(distance.item())

    distance_min = min(distances)
    distance_minidx = distances.index(distance_min)
    return distance_min, distance_minidx
