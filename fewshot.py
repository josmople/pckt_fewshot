
def select_batch(dataset, size, **kwds):
    from torch.utils.data import DataLoader
    for x in DataLoader(dataset, batch_size=size, shuffle=True, **kwds):
        return x


def prototype_fn(x):
    from torch import mean
    return mean(x, dim=0, keepdims=True)


def pairdist_fn(x, y):
    from torch import cdist
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    return cdist(x, y, p=2).squeeze(0)


def episode(*datasets, features_fn, prototype_fn=prototype_fn, pairdist_fn=pairdist_fn, shots=5):
    from torch import cat, arange
    from torch.nn.functional import cross_entropy

    prototypes = []
    queries = []
    for dataset in datasets:
        batch = select_batch(dataset, shots + 1)

        support = batch[:shots]
        support = features_fn(support)
        prototype = prototype_fn(support)
        prototypes.append(prototype)

        del support

        query = batch[shots:]
        query = features_fn(query)
        queries.append(query)

    prototypes = cat(prototypes, dim=0)
    queries = cat(queries, dim=0)
    scores = pairdist_fn(queries, prototypes)
    labels = arange(len(datasets)).to(scores.device)

    loss = cross_entropy(scores, labels)
    loss = loss / len(datasets) / shots
    return loss


def predict(supports, queries, features_fn, prototype_fn=prototype_fn, distance_fn=pairdist_fn):
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
