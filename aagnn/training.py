from .utils import *

def train_step(model, optimizer, features, adj, labels, idx_train, loss_fn):
    model.train()
    optimizer.zero_grad()
    predictions = model(features, adj).squeeze()

    loss = loss_fn(predictions[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return predictions


def random_sample(surrogate_model, features, adj, labels, idx_test, loss_fn, perturbations, k=10):
    min_loss = 1000
    with torch.no_grad():
        for i in range(k):
            sample = torch.bernoulli(perturbations)
            modified_adj = get_modified_adj(adj, sample)

            sample_predictions = surrogate_model(
                features, modified_adj).squeeze()
            loss = loss_fn(
                sample_predictions[idx_test], labels[idx_test])
            if loss < min_loss:
                min_loss = loss
                best = sample

    print(f"Best sample: {int(best.sum())} edges \t Loss: {loss.item():.2f}")

    return best
