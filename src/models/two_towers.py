import torch
import torch.nn as nn
import torch.nn.functional as F

class UserEncoder(nn.Module):
    def __init__(self, num_users, num_cities, embedding_dim = 32):
        super().__init__()
        # sparse Feature Embeddings. mathematically nn.embedding produces the exact same output as pultiplying the one hot enccode with the dense vector, but it bypasses the multiplication step. It is literally just an O(1) srray loos, 
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.city_embedding = nn.Embedding(num_embeddings=num_cities, embedding_dim=embedding_dim)

        concat_dim = (embedding_dim * 2) + 1

        # Multi Layer Perceptron (MLP) for feature fusion.
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64) # Final output dimention space.
        )

    def forward(self, user_id, age, city_id):
        # Etract the geometric coordinates for categorical data
        u_emb = self.user_embedding(user_id)
        c_emb = self.city_embedding(city_id)
        # Age is [batch_size] 
        age = age.unsqueeze(1)

        # Physically fuse the features together
        combined = torch.cat([u_emb, c_emb, age], dim = 1)

        # compress and encode
        return self.mlp(combined)
    
class TwoTowerModel(nn.Module):
    def __init__(self, num_users = 10000, num_cities = 10, embedding_dim = 32):
        super().__init__()
        # Instantiate the 2 tower intepedent neural network,
        self.query_tower = UserEncoder(num_users, num_cities, embedding_dim)
        self.candidate_tower = UserEncoder(num_users, num_cities, embedding_dim)
    def forward(self, batch):
        # Feed the query features into the query Tower
        q_vecs = self.query_tower(
            batch['query_id'],
            batch['query_age'],
            batch['query_city']
        )
        # Feed the candidate features into the candidate tower
        c_vecs = self.candidate_tower(
            batch['candidate_id'],
            batch['candidate_age'],
            batch['candidate_city']
        )
        # L2 Normalizatio : Forces all vectors to have length of one sso the the dot products acts like a cosine similarity.
        q_vecs = F.normalize(q_vecs, p = 2, dim = 1)
        c_vecs = F.normalize(c_vecs, p = 2, dim = 1)

        return q_vecs, c_vecs
    

# Quick validation test
if __name__ == "__main__":
    # Mock data mirroring what the DataLoader outputs
    mock_batch = {
        'query_id': torch.tensor([2456, 436]),
        'query_age': torch.tensor([0.1281, 0.4732]),
        'query_city': torch.tensor([4, 8]),
        'candidate_id': torch.tensor([999, 12]),
        'candidate_age': torch.tensor([0.88, 0.21]),
        'candidate_city': torch.tensor([1, 0])
    }
    
    model = TwoTowerModel()
    q_out, c_out = model(mock_batch)
    print("Query Output Shape:", q_out.shape)
    print("Candidate Output Shape:", c_out.shape)