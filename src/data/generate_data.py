import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_users = 10000, num_matches = 50000, feature_dim = 120):
    """
    Generate synthetic user features and positive interaction pairs.
    Users are represented by normalized vectors. Matches are generated
    by findinng user highcosine similarity.
    """
    np.random.seed(42)
    
    # create the users
    # create an array of user ids from 0 to 9999.
    print("Generating user features...")
    user_ids = np.arange(num_users)
    # Generate a matric of random numbers for their features (10000 rows, 120 columns)
    user_features = np.random.rand(num_users, feature_dim)
    # L2 Normalize to that dot prodect equals cosine similarity.
    user_features = user_features / np.linalg.norm(user_features, axis =1, keepdims= True)
    print("Generating Positive interaction data...")
    queries = []
    candidates = []

    # Generate batches of queries to compute similarities efficiently
    batch_size = 1000
    for _ in range(num_users // batch_size):
        batch_q = np.random.choice(user_ids, batch_size)
        batch_q_feats = user_features[batch_q]
        # compute similarity matrix from their top 50 most similar candidates.
        sim_matrix = np.dot(batch_q_feats, user_features.T)
        # for each query, get the top 50 most similar candidate (excluding itself)
        for i in range(batch_size):
            # argsort sorts ascending, so we take the last 51 and the remove the user themselves
            top_candidates = np.argsort(sim_matrix[i])[-51:]
            top_candidates = top_candidates[top_candidates != batch_q[i]]
            # randomly select one of these highly, similar users as the true match
            match = np.random.choice(top_candidates[-20:])
            queries.append(batch_q[i])
            candidates.append(match)

        
    # save to disk
    os.makedirs("data/raw", exist_ok=True)

    # save user features
    feature_cols = [f"feat_{i}" for i in range(feature_dim)]
    df_users = pd.DataFrame(user_features, columns=feature_cols)
    df_users.insert(0, "user_id", user_ids)
    df_users.to_csv("data/raw/users.csv", index = False)

    # Save Interaction logs
    df_interactions = pd.DataFrame({
        'query_id' : queries,
        'candidate_id' : candidates
    })
    df_interactions.to_csv("data/raw/interactions.csv", index = False)
    print(f"Success. Saved {num_users} users and {num_matches} to data/raw/")

if __name__ == "__main__":
    generate_synthetic_data()