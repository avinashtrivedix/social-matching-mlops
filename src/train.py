import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import the custom dataset and model
from data.dataset import SocialMatchingDataset
from models.two_towers import TwoTowerModel

def Train(epochs = 5):
    # Intilize the data supply.
    print("Initializing the Supply chain...")
    dataset = SocialMatchingDataset(
        interactions_file= "data/raw/interactions.csv",
        features_file= "data/raw/users_features.csv"
        )
    # Batch size 256 means 1 positive pair and 255 -ve pair per query.
    dataloader = DataLoader(dataset, batch_size=256, shuffle = True)
    # Initialize the Architecture & Optimizer.
    print("Initializing the Multi-Modal engine...")
    model = TwoTowerModel(num_users=10000, num_cities=10, embedding_dim=32)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # Standard Cross Entropy Loss serves as the INfoNCE optimizer here.
    loss_fn = nn.CrossEntropyLoss()
    # InfoNCE is not a piece of code; it is a mathematical concept. It is literally just a categorical cross entropy applied to similarity scores. When you calculate the dot product between your queries and candidates, you are genrating a list of similarity score. Cross entropy takes those scores , applies softmax  to turn them into probabilititoes, and forces the true match to 100% while crushing the rest 0%.

    # THe core training loop
    print("starting the trining loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            # 1. Forward Pass (The model extracts IDs, Age, and City directly from the dictionary)
            q_vecs, c_vecs = model(batch)
            
            # 2. InfoNCE Matrix Math
            temperature = 0.05
            logits = torch.matmul(q_vecs, c_vecs.T) / temperature
            
            # 3. Contrastive Targets (The diagonal elements are the true positive pairs)
            labels = torch.arange(logits.size(0)).to(logits.device)
            
            # 4. Backpropagation
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average InfoNCE Loss: {avg_loss:.4f}")
        
    # Save the mathematically sculpted embeddings
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/two_tower_multimodal.pth")
    print("Multi-modal weights successfully saved to artifacts/two_tower_multimodal.pth")

if __name__ == "__main__":
    Train()
