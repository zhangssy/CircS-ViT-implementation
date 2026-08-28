"""
Physics-Guided Coupling Layer (PGCL) for CircS-ViT.

Eqs. (29)--(33) in the manuscript:
    u = [z; theta] in R^{d+K}
    f = FusionMLP(u) in R^{d_h}          (LayerNorm / GELU / Dropout)
    g = sigmoid(MLP_gate(theta)) in R^{d_h}
    v = f odot g
    e = EnhanceMLP(v) in R^{d_h}         (LayerNorm / GELU)
"""

import torch
import torch.nn as nn


class PhysicsGuidedCouplingLayer(nn.Module):
    """Single-branch PGCL: e = PGCL(z, theta)."""

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9, hidden_dim=128, dropout=0.1):
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.physics_guidance = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
    Stage-1 dual-branch wrapper (Algorithm 2).

    Branch 1: linear projection of 12-channel RSD polarimetric features -> z.
    Branch 2: ViT CLS embedding (passed in as cls_features) -> z.
    Both branches share one TopMod_RSD and one PGCL_RSD.
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

        # Branch 1 encoder: linear projection of x_RSD to z in R^d
        self.branch1_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, embed_dim),
        )

        # Shared TopMod_RSD and PGCL_RSD
        self.topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            feature_depth=mlp_feature_depth,
        )
        self.pgcl = PhysicsGuidedCouplingLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Optional L_CE^e head on e (Eq. (23)); shared across branches
        self.e_classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, z):
        """Shared TopMod + PGCL path: z -> (theta, hat_z, e)."""
        theta, hat_z = self.topic_model(z)
        e = self.pgcl(z, theta)
        return theta, hat_z, e

    def forward(self, cls_features, raw_data=None):
        """
        If raw_data is given (Stage 1): run both branches through the shared modules.
        Otherwise (Stage 2 residual / fusion): run only the CLS path.
        Returns e of the CLS path (and topic proportions for logging).
        """
        theta2, _, e2 = self.encode(cls_features)
        if raw_data is None:
            return e2, theta2, theta2

        z1 = self.branch1_projection(raw_data)
        theta1, _, e1 = self.encode(z1)
        return e2, theta1, theta2, e1, z1
