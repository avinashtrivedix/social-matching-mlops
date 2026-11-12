import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users:int, embedding_dim : int = 64):
        super().__init__()
        # Tower_1 : Encodes the query
        self.query_tower = nn.Embedding(num_embeddings=num_users, embedding_dim = embedding_dim)
        # Tower_2 : Encodes the candidates
        self.candidate_tower = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)

    def forward(self, query_ids : torch.Tensor, candidate_ids : torch.Tensor):
        """
        Takes the raw user IDs from the DataLoader and converts the embeddings them into dense vectors.
        """
        query_embeddings = self.query_tower(query_ids)
        candidate_ids = self.candidate_tower(candidate_ids)
        return query_embeddings, candidate_ids
    

# quick test to ensure the architecture ensure the architecturre compiles.
if __name__ == "__main__":
    # Initialze the model for out 10000 users, outputting 64 - dimentional vectors
    model = TwoTowerModel(num_users = 10000, embedding_dim = 64)
    # Simulate the exact output your DataLoader just gave us
    sample_queries = torch.tensor([6553, 4479, 2290, 3440], dtype=torch.long)
    sample_candidates = torch.tensor([7076, 921, 8850, 4984], dtype=torch.long)

    # Pass them through the network
    q_vecs, c_vecs = model(sample_queries, sample_candidates)
    print("Query Embeddings Shape: ", {q_vecs.shape})
    print("candidate Embeddings Shape: ", {c_vecs.shape})