"""
Supervised topic model (sLDA-guided surrogate) for PGCL in CircS-ViT.

Eq. (26):  theta = softmax(MLP_f2t(z)) in Delta^{K-1}
Eq. (27):  hat{z} = Phi^T theta,  L_rec = ||hat{z} - z||_2^2
Eq. (28):  L_topic = L_CE^top + lambda_rec L_rec
           o_top = H^T theta
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

        # Phi in R^{K x d}; hat{z} = Phi^T theta  <=>  theta @ Phi
        self.topic_basis = nn.Parameter(torch.empty(num_topics, embed_dim))
        nn.init.xavier_uniform_(self.topic_basis)

        # MLP_f2t of depth D; last layer has no activation (softmax is applied after).
        # Hidden width is of order d (2d), as in the complexity analysis.
        hidden = 2 * embed_dim
        if feature_depth == 1:
            layers = [nn.Linear(embed_dim, num_topics)]
        elif feature_depth == 2:
            layers = [
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_topics),
            ]
        elif feature_depth == 3:
            layers = [
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_topics),
            ]
        elif feature_depth == 4:
            layers = [
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_topics),
            ]
        else:
            raise ValueError(f"feature_depth D of MLP_f2t must be 1--4, got {feature_depth}")

        self.feature_to_topic = nn.Sequential(*layers)

        # H in R^{K x N_c}; o_top = H^T theta  <=>  theta @ H
        self.eta = nn.Parameter(torch.empty(num_topics, num_classes))
        nn.init.xavier_uniform_(self.eta)

    def forward(self, cls_features):
        topic_logits = self.feature_to_topic(cls_features)
        doc_topic = F.softmax(topic_logits, dim=1)
        reconstructed_features = torch.matmul(doc_topic, self.topic_basis)
        return doc_topic, reconstructed_features

    def compute_supervised_loss(self, cls_features, labels, lambda_rec=0.1):
        """L_CE^top + lambda_rec L_rec (Eqs. (23), (27), (28))."""
        doc_topic, reconstructed_features = self.forward(cls_features)
        rec_loss = torch.sum((reconstructed_features - cls_features) ** 2, dim=1).mean()
        logits = torch.matmul(doc_topic, self.eta)
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss + lambda_rec * rec_loss

    def get_topic_representation(self, cls_features):
        self.eval()
        with torch.no_grad():
            doc_topic, _ = self.forward(cls_features)
        return doc_topic
