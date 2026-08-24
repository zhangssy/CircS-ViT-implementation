"""
Physics-Guided Coupling Layer (PGCL) for CircS-ViT.
Fuses ViT/CLS features with sLDA topic proportions under physics-guided gating.
"""

import torch
import torch.nn as nn


class PhysicsGuidedCouplingLayer(nn.Module):
    """Single-branch Physics-Guided Coupling Layer (PGCL)."""

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9, hidden_dim=128):
        super(PhysicsGuidedCouplingLayer, self).__init__()

        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        fusion_input_dim = embed_dim + num_topics
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.physics_guidance = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, original_features, topic_representation):
        fused_features = torch.cat([original_features, topic_representation], dim=1)
        fused_output = self.fusion_layer(fused_features)
        physics_guidance_signal = self.physics_guidance(topic_representation)
        guided_features = fused_output * physics_guidance_signal
        return self.feature_enhancement(guided_features)

    def get_enhanced_features(self, original_features, topic_representation):
        self.eval()
        with torch.no_grad():
            return self.forward(original_features, topic_representation)


class DualBranchFusionPGCL(nn.Module):
    """
    Dual-branch fusion Physics-Guided Coupling Layer (PGCL).
    During RSD training, both branches process RSD data and are fused into this module.
    During Cir-domain training, this layer accepts CLS features and outputs enhanced features.
    """

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9, hidden_dim=128,
                 dropout=0.1, in_channels=12, img_size=7, mlp_feature_depth=4):
        super(DualBranchFusionPGCL, self).__init__()

        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.img_size = img_size

        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL

        self.branch1_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.branch1_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            feature_depth=mlp_feature_depth,
        )

        self.branch1_pgcl = PhysicsGuidedCouplingLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

        self.branch2_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            feature_depth=mlp_feature_depth,
        )

        self.branch2_pgcl = PhysicsGuidedCouplingLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, cls_features, raw_data=None):
        if raw_data is not None:
            branch1_features = self.branch1_projection(raw_data)
        else:
            branch1_features = cls_features

        branch1_topic, _ = self.branch1_topic_model(branch1_features)
        branch1_enhanced = self.branch1_pgcl(branch1_features, branch1_topic)

        branch2_topic, _ = self.branch2_topic_model(cls_features)
        branch2_enhanced = self.branch2_pgcl(cls_features, branch2_topic)

        fused_features = torch.cat([branch1_enhanced, branch2_enhanced], dim=1)
        enhanced_features = self.fusion_layer(fused_features)
        return enhanced_features, branch1_topic, branch2_topic
