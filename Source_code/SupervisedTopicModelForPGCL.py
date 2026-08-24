"""
Supervised topic model (sLDA-style) for PGCL in CircS-ViT.
Maps continuous features to simplex topic proportions and predicts class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedTopicModelForPGCL(nn.Module):
    """sLDA-guided continuous topic module used by PGCL."""

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9, feature_depth=4):
        super(SupervisedTopicModelForPGCL, self).__init__()

        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.feature_depth = feature_depth

        self.topic_basis = nn.Parameter(torch.randn(num_topics, embed_dim))
        nn.init.xavier_uniform_(self.topic_basis)
        with torch.no_grad():
            self.topic_basis.data = torch.abs(self.topic_basis.data) + 0.1

        if feature_depth == 3:
            self.feature_to_topic = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_topics),
                nn.ReLU(),
            )
        elif feature_depth == 4:
            self.feature_to_topic = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_topics),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"feature_depth must be 3 or 4, got {feature_depth}")

        self.eta = nn.Parameter(torch.randn(num_topics, num_classes))
        nn.init.xavier_uniform_(self.eta)

    def forward(self, cls_features):
        doc_topic = self.feature_to_topic(cls_features)
        doc_topic = F.relu(doc_topic)
        doc_topic = doc_topic / (doc_topic.sum(dim=1, keepdim=True) + 1e-10)
        reconstructed_features = torch.matmul(doc_topic, self.topic_basis)
        return doc_topic, reconstructed_features

    def compute_supervised_loss(self, cls_features, labels):
        doc_topic, reconstructed_features = self.forward(cls_features)
        recon_loss = F.mse_loss(reconstructed_features, cls_features)
        logits = torch.matmul(doc_topic, self.eta)
        ce_loss = F.cross_entropy(logits, labels)

        topic_similarity = torch.matmul(self.topic_basis, self.topic_basis.t())
        mask = torch.eye(self.num_topics, device=topic_similarity.device).bool()
        topic_similarity = topic_similarity.masked_fill(mask, 0)
        diversity_loss = torch.mean(torch.abs(topic_similarity))

        return recon_loss + ce_loss + 0.01 * diversity_loss

    def get_topic_representation(self, cls_features):
        self.eval()
        with torch.no_grad():
            doc_topic, _ = self.forward(cls_features)
        return doc_topic
