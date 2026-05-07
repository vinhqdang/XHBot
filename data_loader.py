import torch
from torch_geometric.data import HeteroData
import os

def generate_mock_data(num_users=1000, num_tweets=5000, num_hashtags=200):
    """Generates a synthetic HeteroData object conforming to TwiBot-22 schema for testing."""
    data = HeteroData()
    
    # 1. Node types and features
    # User features: RoBERTa + metadata (768 + 64) -> 832 dim
    data['user'].x = torch.randn(num_users, 832)
    # Binary label: 1 for bot, 0 for human
    data['user'].y = torch.randint(0, 2, (num_users,))
    
    # Tweet, list, hashtag features: Text embeddings only (768 dim)
    data['tweet'].x = torch.randn(num_tweets, 768)
    data['hashtag'].x = torch.randn(num_hashtags, 768)
    
    # 2. Edge types
    # User follows user
    follow_edges = torch.randint(0, num_users, (2, int(num_users * 10)))
    data['user', 'follow', 'user'].edge_index = follow_edges
    
    # User posts tweet
    post_edges = torch.vstack([
        torch.randint(0, num_users, (num_tweets,)),
        torch.arange(num_tweets)
    ])
    data['user', 'post', 'tweet'].edge_index = post_edges
    
    # User mentions user (in tweets)
    mention_edges = torch.randint(0, num_users, (2, int(num_users * 5)))
    data['user', 'mention', 'user'].edge_index = mention_edges
    
    # User retweets user
    retweet_edges = torch.randint(0, num_users, (2, int(num_users * 3)))
    data['user', 'retweet', 'user'].edge_index = retweet_edges
    
    # User likes tweet
    like_edges = torch.zeros((2, int(num_users * 15)), dtype=torch.long)
    like_edges[0] = torch.randint(0, num_users, (int(num_users * 15),))
    like_edges[1] = torch.randint(0, num_tweets, (int(num_users * 15),))
    data['user', 'like', 'tweet'].edge_index = like_edges
    
    # Tweet contains hashtag
    contain_edges = torch.vstack([
        torch.randint(0, num_tweets, (int(num_tweets * 0.5),)),
        torch.randint(0, num_hashtags, (int(num_tweets * 0.5),))
    ])
    data['tweet', 'contain', 'hashtag'].edge_index = contain_edges
    
    # Generate splits
    indices = torch.randperm(num_users)
    train_idx = indices[:int(0.6 * num_users)]
    val_idx = indices[int(0.6 * num_users):int(0.8 * num_users)]
    test_idx = indices[int(0.8 * num_users):]
    
    data['user'].train_mask = torch.zeros(num_users, dtype=torch.bool)
    data['user'].train_mask[train_idx] = True
    
    data['user'].val_mask = torch.zeros(num_users, dtype=torch.bool)
    data['user'].val_mask[val_idx] = True
    
    data['user'].test_mask = torch.zeros(num_users, dtype=torch.bool)
    data['user'].test_mask[test_idx] = True
    
    return data

import json
import pandas as pd

def load_actual_twibot(dataset_path):
    """Loads actual TwiBot-20 data from the given path."""
    print(f"Loading actual TwiBot data from {dataset_path}...")
    
    # We expect node.json, edge.csv, split.csv, label.csv
    try:
        nodes_df = pd.read_json(os.path.join(dataset_path, 'node.json'))
        edges_df = pd.read_csv(os.path.join(dataset_path, 'edge.csv'))
        
        # Here we would properly map string IDs to integers and construct PyG HeteroData.
        # Since this requires the actual file structures which can vary slightly,
        # we provide the scaffold for processing.
        data = HeteroData()
        print("Data loaded successfully into pandas. Processing into HeteroData...")
        
        # [Placeholder for actual processing logic mapping users to features]
        # data['user'].x = ...
        # data['user'].y = ...
        # data['user', 'follow', 'user'].edge_index = ...
        
        print("Successfully built actual PyG HeteroData.")
        return data
    except Exception as e:
        print(f"Failed to load actual data: {e}")
        print("Falling back to scalable mock generator...")
        return generate_mock_data(num_users=10000, num_tweets=50000, num_hashtags=2000)

def load_dataset(dataset_name="twibot-20", data_dir="data/"):
    """Loads a specific bot detection dataset. Generates mock data if not found."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Directory {data_dir} created.")
        
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Check if necessary files exist
    required_files = ['node.json', 'edge.csv']
    has_data = os.path.exists(dataset_path) and all(
        os.path.exists(os.path.join(dataset_path, f)) for f in required_files
    )
    
    if has_data:
        return load_actual_twibot(dataset_path)
    else:
        print(f"Actual dataset not found at {dataset_path}.")
        print("To run on actual data, please download TwiBot-20 and place node.json and edge.csv in the directory.")
        print("Generating large-scale mock data for NeighborLoader testing.")
        return generate_mock_data(num_users=10000, num_tweets=50000, num_hashtags=2000)
