# Experiment Report

## Task
This project studies node classification on the Cora citation network.

## Models
The following models were implemented and compared:
- MLP
- GCN
- GAT

## Results

| Model | Best Validation Accuracy | Best Test Accuracy |
|------|--------------------------:|-------------------:|
| MLP  | 0.5200 | 0.5230 |
| GCN  | 0.7840 | 0.8060 |
| GAT  | 0.7820 | 0.8040 |

## Analysis
The experimental results show that graph neural networks significantly outperform the MLP baseline. This indicates that graph structure provides crucial relational information for node classification on citation networks.

Among the tested models, GCN achieved the best performance in the current setting, while GAT showed comparable results.

## Conclusion
This project demonstrates the importance of graph structure in node representation learning and provides a reproducible baseline for future exploration, such as text feature fusion and graph-text modeling.
