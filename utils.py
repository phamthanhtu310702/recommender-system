import torch
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
import random

def convert_adj_mat_edge_index_to_r_mat_edge_index(input_edge_index,num_users,num_movies):
    sparse_input_edge_index = SparseTensor(row=input_edge_index[0], 
                                           col=input_edge_index[1], 
                                           sparse_sizes=((num_users + num_movies), num_users + num_movies))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: num_users, num_users :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index

def convert_r_mat_edge_index_to_adj_mat_edge_index(input_edge_index,num_users,num_movies):
    R = torch.zeros((num_users, num_movies)) # 610x9724
    for i in range(len(input_edge_index[0])):
        row_idx = input_edge_index[0][i]
        col_idx = input_edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1) #10333 x 10333
    adj_mat = torch.zeros((num_users + num_movies , num_users + num_movies))
    adj_mat[: num_users, num_users :] = R.clone() #610x9724
    adj_mat[num_users :, : num_users] = R_transpose.clone() #9724x610
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()
    return adj_mat_coo


# helper function for training and compute BPR loss
# since this is a self-supervised learning, we are relying on the graph structure itself and 
# we don't have label other than the graph structure so we need to the folloing function
# which random samples a mini-batch of positive and negative samples

# In link prediction tasks, 
# it is usual to treat existing edges in the graph as positive examples, 
# and non-existent edges as negative examples.
# i.e. in training/prediction you feed the network a subset of all edges in the complete graph, 
# and the associated targets are "this is a real edge" (positive) and "this is not a real edge" (negative).

def sample_mini_batch(batch_size, edge_index):
    """Randomly samples indices of a minibatch given an adjacency matrix

    Args:
        batch_size (int): minibatch size
        edge_index (torch.Tensor): 2 by N list of edges

    Returns:
        tuple: user indices, positive item indices, negative item indices
    """
    # structured_negative_sampling is a pyG library
    # Samples a negative edge :obj:`(i,k)` for every positive edge
    # :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    # tuple of the form :obj:`(i,j,k)`.
    #
    #         >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
    #         ...                               [0, 1, 2, 3]])
    #         >>> structured_negative_sampling(edge_index)
    #         (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))
    edges = structured_negative_sampling(edge_index)
    
    # 3 x edge_index_len
    edges = torch.stack(edges, dim=0)
    
    # here is whhen we actually perform the batch sampe
    # Return a k sized list of population elements chosen with replacement.
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    
    batch = edges[:, indices]
    
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices