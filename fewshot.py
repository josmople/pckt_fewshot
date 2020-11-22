
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


def dist_fn(x, y):
    from torch.nn.functional import pairwise_distance
    return pairwise_distance(x, y, p=2)


def episode(*datasets, features_fn, prototype_fn=prototype_fn, pairdist_fn=pairdist_fn, n_support=5, n_query=100):
    from torch import cat, tensor
    from torch.nn.functional import cross_entropy

    prototypes = []
    for dataset in datasets:
        support = select_batch(dataset, n_support)
        support = features_fn(support)
        prototype = prototype_fn(support)
        assert prototype.size(0) == 1
        prototypes.append(prototype)
    prototypes = cat(prototypes, dim=0)

    total_loss = 0
    for i, dataset in enumerate(datasets):
        queries = select_batch(dataset, n_query)
        queries = features_fn(queries)

        prototype = prototypes[i].unsqueeze(0)
        temp = [1] * (queries.dim() - 1)
        prototype = prototype.repeat(queries.size(0), *temp)

        distances = dist_fn(queries, prototype)
        distance_loss = distances.mean()
        del distances

        scores = pairdist_fn(queries, prototypes)
        labels = tensor([i] * scores.size(0)).to(scores.device)
        class_loss = cross_entropy(scores, labels)
        del scores, labels

        loss = distance_loss + class_loss
        total_loss += loss

    return total_loss


# def episode(*datasets, features_fn, step_fn, n_support=5, n_query=100, prototype_fn=prototype_fn, pairdist_fn=pairdist_fn, dist_fn=dist_fn):
#     from torch import cat, arange
#     from torch.nn.functional import cross_entropy
#     from torch.utils.data import DataLoader

#     dataloaders = []
#     for dataset in datasets:
#         dataloader = DataLoader(dataset, batch_size=n_support + n_query, shuffle=True)
#         dataloader = iter(dataloader)
#         dataloaders.append(dataloader)

#     total_loss = 0
#     for batches in zip(*dataloaders):
#         if any([b.size(0) != n_support + n_query for b in batches]):
#             break

#         prototypes = []
#         queries = []
#         for batch in batches:

#             support = batch[:n_support]
#             support = features_fn(support)
#             prototype = prototype_fn(support)
#             prototypes.append(prototype)

#             del support

#             query = batch[n_query:]
#             query = features_fn(query)
#             queries.append(query)

#         # TODO
#         prototype_batches = cat(prototypes, dim=0)
#         del prototypes
#         queries_batches = cat(queries, dim=0)
#         del queries

#         distances = dist_fn(queries_batches, prototype_batches)
#         distance_loss = distances.mean()
#         del distances

#         scores = pairdist_fn(queries_batches, prototype_batches)
#         labels = arange(len(datasets)).to(scores.device)
#         class_loss = cross_entropy(scores, labels)
#         del scores, labels

#         loss = distance_loss + class_loss
#         total_loss += loss / shots / len(datasets)
#     step_fn(total_loss)


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
