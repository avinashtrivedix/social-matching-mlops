import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the custom dataset and model
from data.dataset import SocialMatchingDataset
from models.two_towers import TwoTowerModel

def Train():
    # Intilize the data supply.
    print("Initializing the Supply chain...")
    dataset = SocialMatchingDataset("data/raw/interactions.csv")
    # Batch size 256 means 1 positive pair and 255 -ve pair per query.
    dataloader = DataLoader(dataset, batch_size=256, shuffle = True)
    # Initialize the Architecture & Optimizer.
    print("Initializing the Model...")
    model = TwoTowerModel(num_users=10000, embedding_dim=64)
    optimizer = optim.Adam(model.parameters(), lr = 0.005)
    # Standard Cross Entropy Loss serves as the INfoNCE optimizer here.
    criterion = nn.CrossEntropyLoss()
    # InfoNCE is not a piece of code; it is a mathematical concept. It is literally just a categorical cross entropy applied to similarity scores. When you calculate the dot product between your queries and candidates, you are genrating a list of similarity score. Cross entropy takes those scores , applies softmax  to turn them into probabilititoes, and forces the true match to 100% while crushing the rest 0%.
    epochs = 5

    # THe core training loop
    print("starting the trining loop...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            queries = batch['query_id']
            candidates = batch['candidate_id']
            # reset the gradient
            optimizer.zero_grad()
            # Forward pass : Get the embedding for the queries and the candidates for this batch
            q_vecs, c_vecs = model(queries, candidates)
            # INFonce
            # compute the similarity matrix between the query and candidate vector
            # shape output : [256,256] grid of compatibility scores
            scores = torch.matmul(q_vecs, c_vecs.T)
            # The correct matches are on the diagonal
            labels = torch.arange(scores.size(0)).to(scores.device)
            #compute loss
            loss = criterion(scores, labels)
            # Backward pass : Calculate how ro move the users in the embedding space to make the correct matches more similar.
            loss.backward()
            # Optimizer : Actually move them
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average InfoNCE Loss : {avg_loss:.4f} ")
    # save the mathematical sculpted embeddings
    torch.save(model.state_dict(), "artifacts/two_tower_model.pth")
    print("Model training complete. Saved the model to artifacts/two_tower_baseline.pth")
if __name__ == "__main__":
    Train()