"""
Vision Transformer backbone for 10-channel Cir-domain patches (Stage 2).

Eqs. (34)--(37): PatchEmbed (P=1) -> CLS + positional encoding ->
L_Cir Transformer blocks, with frozen TopMod/PGCL residual inserts after
layers 0--3 on the CLS token only, then LayerNorm and h_CLS.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
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
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    ViT_Cir. Residual PGCL uses the SAME frozen TopMod/PGCL as Stage-1
    (shared modules), applied only to the CLS token after layers 0--3.
    """

    residual_insert_layers = (0, 1, 2, 3)

    def __init__(self, img_size=7, patch_size=1, in_channels=10, num_classes=9,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=128, dropout=0.1,
                 num_topics=15, use_pgcl=False, rsd_model_path=None,
                 frozen_topic=None, frozen_pgcl=None, pgcl_hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.depth = depth
        self.img_size = img_size
        self.pgcl_hidden_dim = pgcl_hidden_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.pgcl_to_token = nn.Linear(pgcl_hidden_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        if frozen_topic is not None and frozen_pgcl is not None:
            object.__setattr__(self, "_frozen_topic", None)
            object.__setattr__(self, "_frozen_pgcl", None)
            self.set_frozen_guidance(frozen_topic, frozen_pgcl)
        elif rsd_model_path is not None:
            object.__setattr__(self, "_frozen_topic", None)
            object.__setattr__(self, "_frozen_pgcl", None)
            self._load_frozen_guidance(rsd_model_path, embed_dim, num_topics, num_classes)
        else:
            object.__setattr__(self, "_frozen_topic", None)
            object.__setattr__(self, "_frozen_pgcl", None)

    def set_frozen_guidance(self, topic_model, pgcl):
        """Attach shared frozen TopMod/PGCL without registering them as submodules."""
        for p in topic_model.parameters():
            p.requires_grad = False
        for p in pgcl.parameters():
            p.requires_grad = False
        topic_model.eval()
        pgcl.eval()
        object.__setattr__(self, "_frozen_topic", topic_model)
        object.__setattr__(self, "_frozen_pgcl", pgcl)

    def _load_frozen_guidance(self, rsd_model_path, embed_dim, num_topics, num_classes):
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        from PhysicsGuidedCouplingLayer import PhysicsGuidedCouplingLayer

        rsd_state_dict = torch.load(rsd_model_path, map_location="cpu")
        topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
        )
        pgcl = PhysicsGuidedCouplingLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=self.pgcl_hidden_dim,
        )

        topic_keys = {}
        pgcl_keys = {}
        for k, v in rsd_state_dict.items():
            if k.startswith("fusion_pgcl.topic_model."):
                topic_keys[k.replace("fusion_pgcl.topic_model.", "", 1)] = v
            elif k.startswith("fusion_pgcl.pgcl."):
                pgcl_keys[k.replace("fusion_pgcl.pgcl.", "", 1)] = v
            elif k.startswith("fusion_pgcl.branch2_topic_model."):
                topic_keys[k.replace("fusion_pgcl.branch2_topic_model.", "", 1)] = v
            elif k.startswith("fusion_pgcl.branch2_pgcl."):
                pgcl_keys[k.replace("fusion_pgcl.branch2_pgcl.", "", 1)] = v

        if topic_keys:
            topic_model.load_state_dict(topic_keys, strict=False)
        if pgcl_keys:
            pgcl.load_state_dict(pgcl_keys, strict=False)
        print(f"Loaded frozen TopMod/PGCL from {rsd_model_path}")
        self.set_frozen_guidance(topic_model, pgcl)

    @property
    def frozen_topic(self):
        return self._frozen_topic

    @property
    def frozen_pgcl(self):
        return self._frozen_pgcl

    def train(self, mode=True):
        super().train(mode)
        if self._frozen_topic is not None:
            self._frozen_topic.eval()
        if self._frozen_pgcl is not None:
            self._frozen_pgcl.eval()
        return self

    def _apply(self, fn):
        super()._apply(fn)
        if getattr(self, "_frozen_topic", None) is not None:
            self._frozen_topic._apply(fn)
        if getattr(self, "_frozen_pgcl", None) is not None:
            self._frozen_pgcl._apply(fn)
        return self

    def _residual_pgcl_on_cls(self, current_cls):
        """Same frozen TopMod -> PGCL path, then project e back to d."""
        with torch.no_grad():
            theta, _ = self._frozen_topic(current_cls)
            enhanced = self._frozen_pgcl(current_cls, theta)
        return current_cls + self.pgcl_to_token(enhanced)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (
                i in self.residual_insert_layers
                and self._frozen_topic is not None
                and self._frozen_pgcl is not None
            ):
                updated_cls = self._residual_pgcl_on_cls(x[:, 0])
                x = torch.cat([updated_cls.unsqueeze(1), x[:, 1:]], dim=1)

        x = self.norm(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out, cls_out

    def extract_features(self, x):
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
