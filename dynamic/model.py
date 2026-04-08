import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

#defining base directory
BASE_DIR = os.path.expanduser("~/malware/dynamic_ana/td500")

#directories
BENIGN_EMB_DIR = os.path.join(BASE_DIR, "embeddings/benign/nodes")
MALWARE_EMB_DIR = os.path.join(BASE_DIR, "embeddings/malware/nodes")
BENIGN_EDGE_DIR = os.path.join(BASE_DIR, "embeddings/benign/edges")
MALWARE_EDGE_DIR = os.path.join(BASE_DIR, "embeddings/malware/edges")

#output directory for model and results
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dataset(benign_emb_dir, malware_emb_dir, benign_edge_dir, malware_edge_dir):
    
    print("LOADING DATASET") 
    dataset = []
    
    # Load benign samples
    benign_emb_files = sorted(glob.glob(os.path.join(benign_emb_dir, "*.npy"))) # os.path.join looks for files whereas glob.glob returns all the files and then we sort them
    print(f"\nFound {len(benign_emb_files)} benign embedding files") #counts the files we found
    
    for emb_file in benign_emb_files:
        #we get corresponding edge file from node file name as they both have the same names
        base_name = os.path.basename(emb_file)
        edge_file = os.path.join(benign_edge_dir, base_name)
        
        if not os.path.exists(edge_file): #skips if edge file not found nd give a warning to not stop the entire program
            print(f"Warning: No edge file for {base_name}")
            continue
        
        try:
            # Load embeddings and edges
            embeddings = np.load(emb_file)  # [num_nodes, 64]
            edge_index = np.load(edge_file)  # [2, num_edges]
            
            # Create PyG Data object (converts it to data obj for the model)
            data = Data(
                x=torch.FloatTensor(embeddings), # wht each nodes is (embeddings) converted to float
                edge_index=torch.LongTensor(edge_index), #how the nodes connected(edge list) and converted to integers
                y=torch.tensor([0], dtype=torch.long)  # labels for each where 0 is bengin and 1 is malware
            )
            
            dataset.append(data) # appends each dat obj to the dataset 
            #print(f"Loaded benign: {base_name} - Nodes: {embeddings.shape[0]}, Edges: {edge_index.shape[1]}")
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
    
    # Load malware samples and same logic as benign
    malware_emb_files = sorted(glob.glob(os.path.join(malware_emb_dir, "*.npy")))
    print(f"\nFound {len(malware_emb_files)} malware embedding files")
    
    for emb_file in malware_emb_files:
        base_name = os.path.basename(emb_file)
        edge_file = os.path.join(malware_edge_dir, base_name)
        
        if not os.path.exists(edge_file):
            print(f"Warning: No edge file for {base_name}")
            continue
        
        try:
            embeddings = np.load(emb_file)
            edge_index = np.load(edge_file)
            
            data = Data(
                x=torch.FloatTensor(embeddings),
                edge_index=torch.LongTensor(edge_index),
                y=torch.tensor([1], dtype=torch.long)  #label 1 for malware
            )
             
            dataset.append(data)
            #print(f"Loaded malware: {base_name} - Nodes: {embeddings.shape[0]}, Edges: {edge_index.shape[1]}")
        except Exception as e:
                    print(f"Error loading {base_name}: {e}")

        #prints the total count
    print(f"TOTAL LOADED: {len(dataset)} samples")
    benign_count = sum(1 for d in dataset if d.y.item() == 0)
    malware_count = sum(1 for d in dataset if d.y.item() == 1)
    print(f"  Benign:  {benign_count}")
    print(f"  Malware: {malware_count}")
        
    return dataset
        
#gin model
class GIN_Malware_Classifier(nn.Module):
        
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=3, dropout=0.5):
        #initialising function with parameters like hidden_dim and dropout
        super(GIN_Malware_Classifier, self).__init__()#initialised the base pytorch model

        #initialise the hyperparameters
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers (modulelist lets pytorch track the parameters)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim)) #stabilize training using batchnorm

        for _ in range(num_layers - 1):#Hidden layers
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier (final layer)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  #for 2 classes: benign, malware
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        
        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling (graph-level representation)
        x = global_add_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1) #converts to prob

#training
def train_epoch(model, loader, optimizer, device): #training the model for one full pass over the training data
   
    model.train()
    total_loss = 0 #initialise to track
    correct = 0
    total = 0
    
    for batch in loader: # loop through all the graphs
        batch = batch.to(device)
        optimizer.zero_grad() #clears old gradient to prevent accumulation
        out = model(batch) #logs probabilities for each class
        loss = F.nll_loss(out, batch.y) #calculates loss from models output and the true labels
        loss.backward() #backpropogation to calculate gradient
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_preds, all_labels
def train_model(dataset, epochs=100, batch_size=8, learning_rate=0.001):
        
    #print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    #print(f"Total samples: {len(dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    #print("="*70 + "\n")
    
    # Split dataset (80% train, 20% test)
    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=[d.y.item() for d in dataset]
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model = GIN_Malware_Classifier(
        input_dim=64,  # Node2Vec embedding dimension
        hidden_dim=128,
        num_layers=3,
        dropout=0.7
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    
    # Training history
    train_losses = []
    train_accs = []
    test_accs = []
    
    best_test_acc = 0
    best_epoch = 0
    
    #print("="*70)
    print("TRAINING STARTED")
    #print("="*70)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        test_acc, _, _ = evaluate(model, test_loader, device)
        test_accs.append(test_acc)
        
# Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
    
    #print("\n" + "="*70)
    print("TRAINING COMPLETED")
    #print("="*70)
    print(f"Best Test Accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    #print("="*70 + "\n")
    
    # Final evaluation on best model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    
    # Print detailed results
    #print("\n" + "="*70)
    print("FINAL EVALUATION RESULTS")
    #print("="*70)
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=['Benign', 'Malware'],
                                digits=4))
# Save results
    results = {
        'best_test_accuracy': float(best_test_acc),
        'best_epoch': int(best_epoch),
        'final_test_accuracy': float(test_acc),
        'train_losses': [float(x) for x in train_losses],
        'train_accs': [float(x) for x in train_accs],
        'test_accs': [float(x) for x in test_accs],
        'classification_report': classification_report(test_labels, test_preds,
                                                       target_names=['Benign', 'Malware'],
                                                       output_dict=True)
    }
    
    with open(os.path.join(RESULTS_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_DIR}/training_results.json")
    return model, results

def main():
    
    # Load dataset
    dataset = load_dataset(
        BENIGN_EMB_DIR,
        MALWARE_EMB_DIR,
        BENIGN_EDGE_DIR,
        MALWARE_EDGE_DIR
    )
    
    if len(dataset) == 0:
        print(" No data loaded! Check your directories:")
        print(f"  Benign embeddings: {BENIGN_EMB_DIR}")
        print(f"  Malware embeddings: {MALWARE_EMB_DIR}")
        print(f"  Benign edges: {BENIGN_EDGE_DIR}")
        print(f"  Malware edges: {MALWARE_EDGE_DIR}")
        return
    
    # Train model
    model, results = train_model(
        dataset,
        epochs=100,
        batch_size=8,
        learning_rate=0.001
    )
    
    print("training complete")
    print("Model saved to: ~/malware/new_static/models/best_model.pt")  

if __name__ == "__main__":
    main()
