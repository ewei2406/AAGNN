from utils import *

def train_step(model, optimizer, features, adj, labels, idx_train, loss_fn):
    model.train()
    optimizer.zero_grad()
    predictions = model(features, adj).squeeze()

    loss = loss_fn(predictions[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return predictions


def adj_step(surrogate_model, features, adj, labels, idx_train, loss_fn, loss_inverse, perturbations, epoch, lr_0, num_perturbations):
    surrogate_model.eval()
    modified_adj = get_modified_adj(adj, perturbations)

    predictions = surrogate_model(features, modified_adj).squeeze()

    if loss_inverse:
        loss = -loss_fn(predictions[idx_train], labels[idx_train])
    else:
        loss = loss_fn(predictions[idx_train], labels[idx_train])

    adj_grad = torch.autograd.grad(loss, perturbations)[0]

    lr = 200 / (np.sqrt(epoch+1))
    perturbations = perturbations + (lr * adj_grad)
    perturbations = projection(perturbations, num_perturbations)

    print(
        f"Epoch: {epoch+1} \t Edges perturbed: {int(perturbations.sum())} \t Loss: {loss:.2f}")
    
    return perturbations


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
