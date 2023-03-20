import torch
from torch_geometric.nn.conv import MessagePassing
from torch import nn, optim, Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul
class LightGCN(MessagePassing):
    def __init__(self,num_users,
                num_items,
                embedding_dim=64,
                K=3,
                add_self_loops = False):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K # number of layers
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim= self.embedding_dim)# e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim= self.embedding_dim)# e_i^0

        # "Fills the input Tensor with values drawn from the normal distribution"
        # according to LightGCN paper, this gives better performance
        nn.init.normal_(self.users_emb.weight, std =0.1)
        nn.init.normal_(self.items_emb.weight, std =0.1)
    
    def forward(self, edge_index:Tensor):
        """
        Args:
            edge_index (SparseTensor): adjacency matrix
        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        edge_index_norm = gcn_norm(edge_index=edge_index,add_self_loops= self.add_self_loops)

        # embedding layer-th 0 (M+N)xdim
        emb_0 = torch.cat([self.users_emb.weight,self.items_emb.weight])
        embs = [emb_0]

        # emb_k is the emb that we are actually going to push it through the graph layers
        # as described in lightGCN paper formula 7
        emb_k = emb_0

        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x = emb_k, norm = edge_index_norm[1])
            embs.append(emb_k)
        embs = torch.stack(embs, dim=1)

        emb_final = torch.mean(embs, dim=1)

        # splits into e_u^K and e_i^K
        users_emb_final, items_emb_final = torch.split(emb_final, [self.num_users, self.num_items]) 
        
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


