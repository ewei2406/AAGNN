from utils import *

def train_step(model, optimizer, features, adj, labels, idx_train, loss_fn):
    optimizer.zero_grad()
    predictions = model(features, adj).squeeze()

    loss = loss_fn(predictions[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    return predictions
