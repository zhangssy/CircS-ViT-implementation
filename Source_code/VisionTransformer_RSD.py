"""
Vision Transformer backbone for 12-channel RSD polarimetric features (Stage 1).

Branch 1: linear projection of the RSD window to z in R^d.
Branch 2: ViT CLS embedding (L_RSD encoder blocks).
Both branches share TopMod_RSD and PGCL_RSD (Algorithm 2).
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=12, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding via convolution (P=1 is a per-pixel linear map)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Multi-head self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Feed-forward network
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=12, num_classes=9,
                 embed_dim=64, depth=3, num_heads=4, mlp_dim=128, dropout=0.1,
                 num_topics=15, use_pgcl=True, use_dual_branch=True, mlp_feature_depth=4):
        """
        Stage-1 RSD Vision Transformer with optional dual-branch PGCL.

        Args:
            img_size: spatial size of the input window (H_w = W_w)
            patch_size: token patch size P
            in_channels: RSD feature channels (12 in the manuscript)
            num_classes: number of land-cover classes N_c
            embed_dim: embedding dimension d
            depth: number of ViT encoder blocks on Branch 2 (L_RSD)
            num_heads: number of attention heads h
            mlp_dim: feed-forward hidden width
            dropout: dropout rate
            num_topics: number of topics K
            use_pgcl: whether to enable TopMod/PGCL
            use_dual_branch: whether to use the dual-branch encoder (Algorithm 2)
            mlp_feature_depth: depth D of MLP_f2t
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.use_dual_branch = use_dual_branch
        self.img_size = img_size
        self.in_channels = in_channels

        # Delayed imports to avoid circular dependencies
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        from PhysicsGuidedCouplingLayer import PhysicsGuidedCouplingLayer

        self.pgcl_hidden_dim = 128

        if use_dual_branch and use_pgcl:
            # Branch 2: RSD window through the ViT encoder to a CLS embedding
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Positional encoding
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            # Stacked Transformer encoder blocks (depth = L_RSD)
            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

            # Dual-branch wrapper: Branch-1 linear projection + shared TopMod/PGCL
            from PhysicsGuidedCouplingLayer import DualBranchFusionPGCL

            self.fusion_pgcl = DualBranchFusionPGCL(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=self.pgcl_hidden_dim,
                dropout=dropout,
                in_channels=in_channels,
                img_size=img_size,
                mlp_feature_depth=mlp_feature_depth,
            )
            # L_CE^e head is DualBranchFusionPGCL.e_classifier (see pgcl_classifier)

        elif use_pgcl:
            # Single-branch mode (kept for backward compatibility)
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

            self.add_module("topic_model", SupervisedTopicModelForPGCL(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                feature_depth=mlp_feature_depth,
            ))
            self.add_module("pgcl", PhysicsGuidedCouplingLayer(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=self.pgcl_hidden_dim,
            ))
            self.add_module("pgcl_classifier", nn.Sequential(
                nn.LayerNorm(self.pgcl_hidden_dim),
                nn.GELU(),
                nn.Linear(self.pgcl_hidden_dim, num_classes),
            ))
        else:
            # ViT without PGCL
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)

        # Parameter initialization
        if hasattr(self, "pos_embed"):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "cls_token"):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, "head"):
            nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B = x.size(0)

        if self.use_dual_branch and self.use_pgcl:
            # Branch 2: RSD features through the ViT encoder
            x_vit = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x_vit = torch.cat((cls_tokens, x_vit), dim=1)  # (B, 1+num_patches, embed_dim)
            x_vit = x_vit + self.pos_embed
            x_vit = self.pos_drop(x_vit)

            for blk in self.blocks:
                x_vit = blk(x_vit)

            x_vit = self.norm(x_vit)
            cls_features = x_vit[:, 0]  # CLS token (B, embed_dim)

            # Shared TopMod/PGCL (Algorithm 2); logits from Branch-2 e
            outputs = self.fusion_pgcl(cls_features, raw_data=x)
            e2 = outputs[0]
            branch2_topic = outputs[2]
            logits = self.pgcl_classifier(e2)
            return logits, cls_features, branch2_topic

        elif self.use_pgcl:
            # Single-branch mode (kept for backward compatibility)
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # CLS token

            topic_representation, _ = self.topic_model(cls_out)  # (B, num_topics)
            enhanced_features = self.pgcl(cls_out, topic_representation)  # (B, hidden_dim)
            logits = self.pgcl_classifier(enhanced_features)
            return logits, cls_out, topic_representation
        else:
            # ViT without PGCL
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # CLS token

            out = self.head(cls_out)
            return out, cls_out

    def extract_features(self, x):
        """
        Extract the CLS-token embedding.

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            CLS features of shape (B, embed_dim)
        """
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        return cls_out

    def get_topic_representation(self, x):
        """
        Return simplex topic proportions (for analysis).

        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            topic proportions of shape (B, num_topics)
        """
        if not self.use_pgcl:
            raise ValueError("PGCL is disabled; topic representation is unavailable.")

        cls_features = self.extract_features(x)
        topic_representation = self.topic_model.get_topic_representation(cls_features)
        return topic_representation

    @property
    def topic_model(self):
        if self.use_dual_branch and self.use_pgcl:
            return self.fusion_pgcl.topic_model
        return self._modules.get("topic_model")

    @property
    def pgcl(self):
        if self.use_dual_branch and self.use_pgcl:
            return self.fusion_pgcl.pgcl
        return self._modules.get("pgcl")

    @property
    def pgcl_classifier(self):
        if self.use_dual_branch and self.use_pgcl:
            return self.fusion_pgcl.e_classifier
        return self._modules.get("pgcl_classifier")
