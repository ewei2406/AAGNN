from .utils import *

def train_step(model, optimizer, features, adj, labels, idx_train, loss_fn, iterator=None):
    model.train()
    optimizer.zero_grad()
    predictions = model(features, adj).squeeze()

    loss = loss_fn(predictions[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    if iterator:
        iterator.set_postfix({"loss": f"{loss:.2f}"})

    return predictions