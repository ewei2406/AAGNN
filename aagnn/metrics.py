import numpy as np
import torch.nn.functional as F
import torch
# from .utils import to_index



def acc(predictions, labels):
    correct = (predictions.argmax(1) == labels).sum()
    acc = correct / predictions.size(dim=0)
    return acc



# def show_acc(epoch, predictions, labels, idx_train, idx_test, verbose=False, prefix=""):
#     train_correct = (predictions.argmax(
#         1)[idx_train] == labels[idx_train]).sum()
#     train_acc = (train_correct) / (idx_train.sum())

#     test_correct = (predictions.argmax(1)[idx_test] == labels[idx_test]).sum()
#     test_acc = (test_correct) / (idx_test.sum())

#     loss = F.cross_entropy(predictions[idx_train], labels[idx_train])

#     if verbose:
#         if epoch:
#             print(
#                 f"{prefix}Epoch: {epoch} \t Train: {train_acc:.2%} \t Test: {test_acc:.2%} \t Loss: {loss:.2f}")
#         else:
#             print(
#                 f"{prefix}Train: {train_acc:.2%} \t Test: {test_acc:.2%} \t Loss: {loss:.2f}")
    
#     return train_acc, test_acc


# def evaluate_acc(model, features, adj, labels, idx_train, idx_test, prefix=""):
#     predictions = model(features, adj).squeeze()
#     return show_acc(False, predictions, labels, idx_train, idx_test, True, prefix)


# def edge_types(adj, labels):
#     edges = to_index(adj).t().squeeze()
#     similarities = []
#     for edge in edges:
#         is_same = labels[edge[0]] == labels[edge[1]]
#         similarities.append(is_same)
    
#     similarities = np.array(similarities)

#     return int(similarities.sum()), int((len(similarities) - similarities.sum()))


# def summarize_edges (adj, perturbed, labels):
#     same, diff = edge_types(adj, labels)
#     p_same, p_diff = edge_types(perturbed, labels)

#     print(f"")
#     print(f"Edge Summary")
#     print(f"        Same\tDiff")
#     print(f"      +------------------")
#     print(f"Clean | {same}\t{diff}")
#     print(f"Atk   | {p_same}\t{p_diff}")
#     print(f"Diff  | {p_same - same}\t{p_diff - diff}")
#     print(f"      +------------------")


# def same_diff(edges, labels):
#     similarities = []
#     for edge in edges:
#         is_same = labels[edge[0]] == labels[edge[1]]
#         similarities.append(is_same)

#     similarities = np.array(similarities)
#     return int(similarities.sum()), int((len(similarities) - similarities.sum()))


# def show_change_matrix(adj, perturbed, labels):
#     diff = perturbed - adj
#     added = to_index(diff.clamp(0, 1)).t()
#     removed = to_index(diff.clamp(-1, 0).abs()).t()

#     s_added, d_added = same_diff(added, labels)
#     s_removed, d_removed = same_diff(removed, labels)

#     total = s_added + d_added + s_removed + d_removed

#     print(f"")
#     print(f"Changes Applied")
#     print(f"          Same\tDiff")
#     print(f"        +---------------")
#     print(f"Added   | {s_added}\t{d_added}")
#     print(f"Removed | {s_removed}\t{d_removed}")
#     print(f"        +---------------")
#     print(f"Total Chages={total}")


# def acc_by_label(predictions, labels, idx_test, targets, verbose=True):
#     """
#     Returns the mean accuracy of predictions on specified label targets
    
#     Parameters
#     ---
#     predictions : torch.tensor
#         Model predictions
#     labels : torch.tensor
#         Ground-truth labels
#     idx_test : torch.tensor
#         Test indices
#     targets : array
#         Target labels
#     verbose : bool (optional)
#         Enable displaying result

#     Returns
#     ---
#     out : float
#         Accuracy (0, 1)
    
#     Examples
#     ---
    
#     """

#     cumulative = 0

#     for l in targets:
#         target_labels = labels[idx_test] == l

#         test_correct = (predictions.argmax(1)[idx_test][target_labels] == l).sum()
#         test_acc = test_correct / (target_labels.sum())

#         cumulative += test_acc

#     return (cumulative / len(targets)).item()


# def count_label_edges(adj, labels, l0, l1):
#     """
#     Returns the number of edges that are between nodes of label l0 and l1 in the graph.
    
#     Parameters
#     ---
#     adj : Torch.tensor
#         Binary adjacency matrix
#     labels : Array-like
#         Labels of nodes
#     l0 : any
#         Target label of l0
#     l1 : any
#         Target label of l1
    
#     Returns
#     ---
#     out : int
#         Number of edges of type l0-l1
    
#     Examples
#     ---
#     >>>
    
#     """
    
#     edges = to_index(adj).t().squeeze()
#     matches = []
#     for edge in edges:
#         match = (labels[edge[0]] == l0 and labels[edge[1]] == l1) or (
#             labels[edge[0]] == l1 and labels[edge[1]] == l0)
#         matches.append(match)

#     matches = np.array(matches)

#     return int(matches.sum())


# def target_diff(edges, labels, target_idx):
#     similarities = []
#     differences = []
#     exclusions = []
#     for edge in edges:
#         similar = edge[0] in target_idx and edge[1] in target_idx
#         similarities.append(similar)

#         diff = (edge[0] in target_idx and edge[1] not in target_idx) or (
#             edge[0] not in target_idx and edge[1] in target_idx)
#         differences.append(diff)

#         exclude = edge[0] not in target_idx and edge[1] not in target_idx
#         exclusions.append(exclude)

#     similarities = np.array(similarities)

#     def count(a):
#         return int(np.array(a).sum())

#     return count(similarities), count(differences), count(exclusions)


# def show_target_change_matrix(adj, perturbed, labels, target_idx):
#     diff = perturbed - adj
#     added = to_index(diff.clamp(0, 1)).t()
#     removed = to_index(diff.clamp(-1, 0).abs()).t()

#     s_added, d_added, e_added = target_diff(added, labels, target_idx)
#     s_removed, d_removed, e_added = same_diff(removed, labels, target_idx)

#     total = s_added + d_added + e_added + s_removed + d_removed + e_added

#     print(f"")
#     print(f"Changes Applied")
#     print(f"           Added\tRemoved")
#     print(f"          +---------------")
#     print(f" G0- G0   | {s_added}\t{s_removed}")
#     print(f" G0-!G0   | {d_added}\t{d_removed}")
#     print(f"!G0-!G0   | {e_added}\t{e_added}")
#     print(f"          +---------------")
#     print(f"Total Chages={total}")
