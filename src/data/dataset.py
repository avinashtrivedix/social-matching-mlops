import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SocialMatchingDataset(Dataset):
    def __init__(self, interactions_file: str, features_file: str):
        # 1. Load the interaction signals
        self.interactions = pd.read_csv(interactions_file)
        
        # 2. Load features and set user_id as the index for O(1) mathematical lookup time
        self.features = pd.read_csv(features_file).set_index('user_id')
        
    def __len__(self):
        return len(self.interactions)
        
    def __getitem__(self, idx: int) -> dict:
        row = self.interactions.iloc[idx]
        
        q_id = int(row['query_id'])
        c_id = int(row['candidate_id'])
        
        # Pull the dense and sparse features for the query user
        q_feats = self.features.loc[q_id]
        
        # Pull the dense and sparse features for the candidate user
        c_feats = self.features.loc[c_id]
        
        return {
            # Query Tensors
            "query_id": torch.tensor(q_id, dtype=torch.long),
            "query_age": torch.tensor(q_feats['age_norm'], dtype=torch.float32),
            "query_city": torch.tensor(int(q_feats['city_id']), dtype=torch.long),
            
            # Candidate Tensors
            "candidate_id": torch.tensor(c_id, dtype=torch.long),
            "candidate_age": torch.tensor(c_feats['age_norm'], dtype=torch.float32),
            "candidate_city": torch.tensor(int(c_feats['city_id']), dtype=torch.long)
        }

# Quick validation test
if __name__ == "__main__":
    dataset = SocialMatchingDataset(
        interactions_file="data/raw/interactions.csv", 
        features_file="data/raw/users_features.csv"
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    print("Query IDs:   ", batch['query_id'])
    print("Query Ages:  ", batch['query_age'])
    print("Query Cities:", batch['query_city'])