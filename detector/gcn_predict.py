import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool as gmp
from torch_geometric.datasets import UPFD
from torch_geometric.transforms import ToUndirected
import torch_geometric.transforms as T
import sys
import os

class GCN(torch.nn.Module):
    def __init__(self, args):
        assert args['num_layers'] >= 2, "num_layers must be >= 2."
        super(GCN, self).__init__()
        self.num_layers = args['num_layers']
        self.dropout = args['dropout']
        self.convs = torch.nn.ModuleList([GCNConv(args['num_features'], args['hidden_dim'])])
        self.bns = torch.nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.bns.extend([torch.nn.BatchNorm1d(args['hidden_dim'])])
            self.convs.extend([GCNConv(args['hidden_dim'], args['hidden_dim'])])
        self.lin0 = Linear(args['hidden_dim'], args['num_classes'])

    def forward(self, data):
        out, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers - 1):
            out = self.convs[i](out, edge_index)
            out = self.bns[i](out)
            out = F.relu(out)
            if self.dropout > 0:
                out = F.dropout(out, training=self.training)
        out = self.convs[self.num_layers - 1](out, edge_index)
        out = gmp(out, batch)
        out = self.lin0(out)
        out = F.log_softmax(out, dim=-1)
        return out

def load_data(split, feature=None):
    max_nodes = 500
    if feature == 'content':
        return UPFD('/tmp/test', "politifact", feature, split, transform=T.ToDense(max_nodes), pre_transform=ToUndirected())
    else:
        data_profile = UPFD('/tmp/test', "politifact", "profile", split, transform=T.ToDense(max_nodes), pre_transform=ToUndirected())
        data_bert = UPFD('/tmp/test', "politifact", "bert", split, transform=T.ToDense(max_nodes), pre_transform=ToUndirected())
        data_profile.data.x = torch.cat((data_profile.data.x, data_bert.data.x), dim=1)
        return data_profile

def train_and_save():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data('train')
    val_dataset = load_data('val')
    args = {
        'num_features': dataset.num_features,
        'hidden_dim': 64,
        'num_classes': dataset.num_classes,
        'dropout': 0.5,
        'num_layers': 2
    }
    model = GCN(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_acc = 0
    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.data.to(device))
        loss = F.nll_loss(out, dataset.data.y)
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_dataset.data.to(device))
            pred = val_out.argmax(dim=1)
            correct = int((pred == val_dataset.data.y).sum())
            val_acc = correct / val_dataset.data.y.size(0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'gcn_model.pt'))
    print("Training complete. Best val acc:", best_val_acc)

def predict(tweet_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = load_data('test')
    args = {
        'num_features': test_dataset.num_features,
        'hidden_dim': 64,
        'num_classes': test_dataset.num_classes,
        'dropout': 0.5,
        'num_layers': 2
    }
    model = GCN(args).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'gcn_model.pt')
    if not os.path.exists(model_path):
        print('Model not found. Please train the model first.')
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # For demonstration, use the tweet_id as an index into the test set
    idx = int(tweet_id) % len(test_dataset)
    data = test_dataset[idx].to(device)
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).item()
    print(pred)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        train_and_save()
    elif len(sys.argv) == 2:
        predict(sys.argv[1])
    else:
        print("Usage: python gcn_predict.py <tweet_id> or python gcn_predict.py train")