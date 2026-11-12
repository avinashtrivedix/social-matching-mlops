import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SocialMatchingDataset(Dataset):
    def __init__(self, interactions_file: str):
        """
        Loads the Positive interaction for the 2 tower model.
        In - batch negative sampling handles the negatives dynamically during training.
        """
        # Load the interaction CSV into memry.
        self.interaction = pd.read_csv(interactions_file)

        # Ensure we hava the correct columns (adjust names based on your generation script)
        if 'query_id' not in self.interaction.columns or 'candidate_id' not in self.interaction.columns:
            raise ValueError("INteractions CSV must contain 'query_id' and 'candidate_id' columns.")
        
    def __len__(self):
        """Returns the total number of positive interactions."""
        return len(self.interaction)
    
    def __getitem__(self, idx : int) -> dict:
        """Fetches a single interaction pair and converts it to a PyTorch tensor."""
        row = self.interaction.iloc[idx]
        return {
            "query_id" : torch.tensor(row['query_id'], dtype = torch.long),
            "candidate_id" : torch.tensor(row['candidate_id'], dtype = torch.long)
        }
    

# quick test to ensure that the architecture works locally
if __name__ == "__main__":
    #Point to the data just versione with DVC
    dataset = SocialMatchingDataset(interactions_file="data/raw/interactions.csv")
    # create the DataLoader to feed the model in batches.
    dataloader  = DataLoader(dataset, batch_size=4, shuffle=True)
    # Fetch a single batch to verify
    sample_batch = next(iter(dataloader))
    print("Query Batch:", sample_batch['query_id'])
    print("Candidate batch:", sample_batch['candidate_id'])