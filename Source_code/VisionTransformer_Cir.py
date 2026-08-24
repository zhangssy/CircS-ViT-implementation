"""
Vision Transformer backbone for circular-polarization (Cir) feature maps.
PGCL layers from the pre-trained RSD stage can be injected after each encoder block.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding via 1x1 convolution
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=108, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=10, num_classes=9,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=108, dropout=0.1,
                 num_topics=15, use_pgcl=False, rsd_model_path=None):
        """
        Cir-domain Vision Transformer.

        Args:
            img_size: spatial size of the input patch
            in_channels: number of Cir feature channels
            num_classes: number of land-cover classes
            embed_dim: token embedding dimension
            depth: number of transformer encoder blocks
            rsd_model_path: optional checkpoint from Stage-1 (RSD PGCL) training
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.depth = depth

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # PGCL guidance modules loaded from the RSD checkpoint (one per encoder block)
        self.rsd_pgcl_layers = nn.ModuleList()
        if rsd_model_path is not None:
            self._load_rsd_pgcl_layers(rsd_model_path, embed_dim, num_topics, num_classes)

        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def _load_rsd_pgcl_layers(self, rsd_model_path, embed_dim, num_topics, num_classes):
        """Load PGCL guidance weights from the Stage-1 RSD checkpoint."""
        try:
            rsd_state_dict = torch.load(rsd_model_path, map_location='cpu')

            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedCouplingLayer import PhysicsGuidedCouplingLayer
            from PhysicsGuidedCouplingLayer import DualBranchFusionPGCL

            fusion_pgcl_keys = {k.replace('fusion_pgcl.', ''): v
                                for k, v in rsd_state_dict.items()
                                if 'fusion_pgcl' in k}

            if fusion_pgcl_keys:
                fusion_pgcl = DualBranchFusionPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128,
                    dropout=0.1,
                    in_channels=10,  # Cir feature channels (may differ from RSD)
                    img_size=7
                )

                fusion_pgcl_dict = fusion_pgcl.state_dict()
                filtered_fusion_keys = {}
                skipped_keys = []

                for k, v in fusion_pgcl_keys.items():
                    if k in fusion_pgcl_dict:
                        if fusion_pgcl_dict[k].shape == v.shape:
                            filtered_fusion_keys[k] = v
                        else:
                            skipped_keys.append(
                                f"{k}: checkpoint shape {v.shape} != model shape {fusion_pgcl_dict[k].shape}"
                            )
                    else:
                        skipped_keys.append(f"{k}: key not in model")

                if skipped_keys:
                    print("  Warning: skipped mismatched PGCL keys (first 5 shown):")
                    for key in skipped_keys[:5]:
                        print(f"    - {key}")
                    if len(skipped_keys) > 5:
                        print(f"    ... and {len(skipped_keys) - 5} more")
                    print("  Note: branch1_projection may differ between RSD and Cir channels.")

                fusion_pgcl.load_state_dict(filtered_fusion_keys, strict=False)
                print("  Loaded dual-branch fusion PGCL weights.")

                for _ in [0, 1, 2, 3]:
                    projection = nn.Linear(128, embed_dim)
                    pgcl_module = nn.ModuleDict({
                        'fusion_pgcl': fusion_pgcl,
                        'projection': projection
                    })
                    self.rsd_pgcl_layers.append(pgcl_module)
            else:
                print("  Fusion PGCL not found; falling back to single-branch PGCL.")
                for _ in [0, 1, 2, 3]:
                    topic_model = SupervisedTopicModelForPGCL(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes
                    )
                    pgcl = PhysicsGuidedCouplingLayer(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes,
                        hidden_dim=128
                    )
                    projection = nn.Linear(128, embed_dim)

                    topic_model_keys = {k.replace('topic_model.', ''): v
                                       for k, v in rsd_state_dict.items()
                                       if 'topic_model' in k and 'fusion_pgcl' not in k}
                    pgcl_keys = {k.replace('pgcl.', ''): v
                                 for k, v in rsd_state_dict.items()
                                 if 'pgcl' in k and 'fusion_pgcl' not in k}

                    if topic_model_keys:
                        topic_model.load_state_dict(topic_model_keys, strict=False)
                    if pgcl_keys:
                        pgcl.load_state_dict(pgcl_keys, strict=False)

                    pgcl_module = nn.ModuleDict({
                        'topic_model': topic_model,
                        'pgcl': pgcl,
                        'projection': projection
                    })
                    self.rsd_pgcl_layers.append(pgcl_module)

            print(f"Successfully loaded RSD PGCL layers from {rsd_model_path}")
        except Exception as e:
            print(f"Warning: Failed to load RSD PGCL layers: {e}")
            print("Creating new PGCL layers instead...")
            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedCouplingLayer import PhysicsGuidedCouplingLayer

            for _ in [0, 1, 2, 3]:
                topic_model = SupervisedTopicModelForPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes
                )
                pgcl = PhysicsGuidedCouplingLayer(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128
                )
                projection = nn.Linear(128, embed_dim)
                pgcl_module = nn.ModuleDict({
                    'topic_model': topic_model,
                    'pgcl': pgcl,
                    'projection': projection
                })
                self.rsd_pgcl_layers.append(pgcl_module)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in [0, 1, 2, 3] and len(self.rsd_pgcl_layers) > 0:
                current_cls = x[:, 0]
                pgcl_idx = i
                if pgcl_idx < len(self.rsd_pgcl_layers):
                    rsd_module = self.rsd_pgcl_layers[pgcl_idx]

                    if 'fusion_pgcl' in rsd_module:
                        enhanced_features, _, _ = rsd_module['fusion_pgcl'](
                            current_cls, raw_data=None  # Cir stage uses CLS features only
                        )
                    else:
                        topic_representation, _ = rsd_module['topic_model'](current_cls)
                        enhanced_features = rsd_module['pgcl'](current_cls, topic_representation)

                    projected_features = rsd_module['projection'](enhanced_features)
                    updated_cls = current_cls + projected_features
                    x = torch.cat([updated_cls.unsqueeze(1), x[:, 1:]], dim=1)

        x = self.norm(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out, cls_out

    def extract_features(self, x):
        """Return CLS token features for downstream analysis."""
        B = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def get_topic_representation(self, x):
        """Return topic proportions when PGCL is enabled."""
        if not self.use_pgcl:
            raise ValueError("PGCL is disabled; topic representation is unavailable.")

        cls_features = self.extract_features(x)
        return self.topic_model.get_topic_representation(cls_features)
