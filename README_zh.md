[English](./README.md) | [中文](./README_zh.md)

# 图文本节点分类项目

本项目基于图学习方法，在 Cora 引文网络数据集上进行节点分类任务实践。

## 项目概述

本项目的目标是复现几个经典的节点分类基线模型，包括：
- MLP
- GCN
- GAT

本项目也作为一个小型、可复现的图学习实践项目，用于理解图神经网络在节点分类任务中的基本作用。

## 数据集

本项目使用 PyTorch Geometric 提供的 **Cora** 引文网络数据集。

- **节点（Nodes）**：学术论文
- **边（Edges）**：论文之间的引用关系
- **节点特征（Node Features）**：词袋表示
- **标签（Labels）**：论文所属类别

## 项目动机

现实世界中的许多数据天然具有关系结构，例如引文网络、社交网络、推荐系统和知识图谱。图学习方法能够同时利用节点属性和结构信息。本项目希望通过一个小型但可复现的实验流程，理解图神经网络的基本价值与作用。

## 模型说明

本项目包含以下模型：
- **MLP**：不利用图结构、仅使用节点特征的基线模型
- **GCN**：图卷积网络
- **GAT**：图注意力网络

## 项目结构

    graph-text-node-classification/
    ├─ README.md
    ├─ README_zh.md
    ├─ requirements.txt
    ├─ train.py
    ├─ models.py
    ├─ utils.py
    ├─ report.md
    ├─ plot_results.py
    ├─ results/
    │  ├─ metrics.csv
    │  └─ comparison.png

## 实验结果

以下是在 Cora 数据集上得到的实验结果：

| 模型 | 最优验证集准确率 | 最优测试集准确率 |
|------|------------------:|-----------------:|
| MLP  | 0.5200 | 0.5230 |
| GCN  | 0.7840 | 0.8060 |
| GAT  | 0.7820 | 0.8040 |

## 结果可视化

![Model Comparison on Cora](results/comparison.png)

## 关键结论

- MLP 基线模型的效果明显弱于图神经网络。
- 在 Cora 数据集上，GCN 和 GAT 都显著优于 MLP。
- 在当前实验设置下，GCN 取得了最好的结果，GAT 表现与其接近。
- 这些结果表明，图结构信息对节点分类任务非常重要。

## 运行方式

安装依赖：

    pip install -r requirements.txt

运行训练：

    python train.py --model mlp
    python train.py --model gcn
    python train.py --model gat

绘制结果图：

    python plot_results.py

## 后续工作

后续可以进一步探索文本特征融合，将文本语义表示与图结构特征结合起来，用于节点分类任务。
