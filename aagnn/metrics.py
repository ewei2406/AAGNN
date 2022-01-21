import numpy as np
import torch.nn.functional as F
import torch
from . import utils



def acc(predictions, labels):
    correct = (predictions.argmax(1) == labels).sum()
    acc = correct / predictions.size(dim=0)
    return acc.item()

def mask_adj(adj, bool_list):
    idx = utils.bool_to_idx(bool_list).squeeze()

    temp_adj = adj.clone()
    temp_adj.index_fill_(dim=0, index=idx, value=0)
    diff = adj - temp_adj

    temp_adj = diff.clone()
    temp_adj.index_fill_(dim=1, index=idx, value=0)
    diff = diff - temp_adj

    # add = int(diff.clamp(0,1).sum() / 2)
    # remove = int(diff.clamp(-1,0).abs().sum() / 2)

    return diff



def show_metrics(changes, labels, g0):
    print("METRICS===================")

    def print_same_diff(type, adj):
        edges = utils.to_edges(adj)
        same = 0
        for edge in edges.t():
            same += int(labels[edge[0]].item() == labels[edge[1]].item())
        
        diff = edges.shape[1] - same

        print(f"{type} | {int(same)}  \t{int(diff)}  \t{int(same+diff)}")


    def print_add_remove(adj):
        add = adj.clamp(0,1)
        remove = adj.clamp(-1,0).abs()
        print("      SAME \tDIFF\tTOTAL")
        print("    +----------------------------")
        print_same_diff("ADD", add)
        print_same_diff("REM", remove)
        print("    +----------------------------")
    # print_add_remove(changes)

    print("Edge type: G0-G0 =====")
    g0_adj = mask_adj(changes, g0)
    print_add_remove(g0_adj)

    print("Edge type: GX-GX =====")
    gX_adj = mask_adj(changes, ~g0)
    print_add_remove(gX_adj)

    print("Edge type: G0-GX =====")
    g0gX_adj = (changes - g0_adj - gX_adj)
    print_add_remove(g0gX_adj)

    print_same_diff("TOT", changes)


    # total_add = changes.clamp(0,1).sum() / 2
    # total_remove = changes.clamp(-1,0).abs().sum() / 2

    # edges = to_edges(changes).t()

    # total_same = 0

    # for edge in edges:
    #     same = labels[edge[0]].item() == labels[edge[1]].item()

    #     n1_isg0 = g0[edge[0]]
    #     n2_isg0 = g0[edge[1]]

    #     if same:
    #         total_same += 1

    #     if n1_isg0 and n2_isg0:
            

    # print(total_same)

    # print(f"Add: {total_add:0}\t Remove: {total_remove:0}")
    # print(edges.shape[1])
    # print(labels)
    # print(g0)

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
