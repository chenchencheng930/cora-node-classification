import os
import csv
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models import MLP, GCN, GAT


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy(pred, y):
    return (pred == y).sum().item() / y.size(0)


def evaluate(model, data, device):
    model.eval()
    with torch.no_grad():
        if isinstance(model, MLP):
            out = model(data.x.to(device))
        else:
            out = model(data.x.to(device), data.edge_index.to(device))

        pred = out.argmax(dim=1)

        train_acc = accuracy(
            pred[data.train_mask],
            data.y[data.train_mask].to(device)
        )
        val_acc = accuracy(
            pred[data.val_mask],
            data.y[data.val_mask].to(device)
        )
        test_acc = accuracy(
            pred[data.test_mask],
            data.y[data.test_mask].to(device)
        )

    return train_acc, val_acc, test_acc


def train_one_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    if isinstance(model, MLP):
        out = model(data.x.to(device))
    else:
        out = model(data.x.to(device), data.edge_index.to(device))

    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask].to(device)
    )
    loss.backward()
    optimizer.step()
    return loss.item()


def save_result(args, best_val_acc, best_test_acc):
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", "metrics.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "hidden_dim", "lr", "weight_decay",
                "epochs", "best_val_acc", "best_test_acc"
            ])
        writer.writerow([
            args.model,
            args.hidden_dim,
            args.lr,
            args.weight_decay,
            args.epochs,
            round(best_val_acc, 4),
            round(best_test_acc, 4),
        ])


def build_model(args, dataset):
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes

    if args.model == "mlp":
        model = MLP(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=out_channels,
            dropout=args.dropout,
        )
    elif args.model == "gcn":
        model = GCN(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=out_channels,
            dropout=args.dropout,
        )
    elif args.model == "gat":
        model = GAT(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=out_channels,
            heads=args.heads,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gcn", "gat"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = Planetoid(root="data/Planetoid", name="Cora")
    data = dataset[0]

    print(data)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Number of node features:", data.num_node_features)
    print("Number of classes:", dataset.num_classes)

    model = build_model(args, dataset).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, data, optimizer, device)
        train_acc, val_acc, test_acc = evaluate(model, data, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch: {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")

    save_result(args, best_val_acc, best_test_acc)


if __name__ == "__main__":
    main()
