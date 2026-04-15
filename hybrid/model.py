import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.expanduser("~/malware/hybrid/t11")
BENIGN_EMB_DIR = os.path.join(BASE_DIR, "benign/nodes")
MALWARE_EMB_DIR = os.path.join(BASE_DIR, "malware/nodes")
BENIGN_EDGE_DIR = os.path.join(BASE_DIR, "benign/edges")
MALWARE_EDGE_DIR = os.path.join(BASE_DIR, "malware/edges")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_DIM = 128  
HIDDEN_DIM = 128
BATCH_SIZE = 16  
EPOCHS = 100

#dataset loading
def load_dataset():
    dataset = []
    configs = [(BENIGN_EMB_DIR, BENIGN_EDGE_DIR, 0), (MALWARE_EMB_DIR, MALWARE_EDGE_DIR, 1)]
    
    for emb_dir, edge_dir, label in configs:
        emb_files = sorted(glob.glob(os.path.join(emb_dir, "*.npy")))
        for emb_file in emb_files:
            fname = os.path.basename(emb_file)
            edge_file = os.path.join(edge_dir, fname)
            
            if os.path.exists(edge_file):
                dataset.append(Data(
                    x=torch.FloatTensor(np.load(emb_file)),
                    edge_index=torch.LongTensor(np.load(edge_file)),
                    y=torch.tensor([label])
                ))
    print(f"Loaded {len(dataset)} synchronized hybrid samples.")
    return dataset

model arch
class HybridGIN(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(HybridGIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        # Global Pooling (Mean + Max)
        x_pool = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return F.log_softmax(self.classifier(x_pool), dim=1)


dataset = load_dataset()
train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=[d.y.item() for d in dataset])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridGIN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(batch), batch.y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=1)
                correct += (pred == batch.y).sum().item()
        print(f"Epoch {epoch} | Accuracy: {correct/len(test_data):.4f}")


#for final report
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        y_pred.extend(model(batch).argmax(dim=1).cpu().numpy())
        y_true.extend(batch.y.cpu().numpy())

print("final report")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malware']))
