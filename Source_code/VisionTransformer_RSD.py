import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=12, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d 实现 patch embedding
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
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=12, num_classes=9,
                 embed_dim=64, depth=3, num_heads=4, mlp_dim=128, dropout=0.1,
                 num_topics=15, use_pgcl=True, use_dual_branch=True, mlp_feature_depth=4):
        """
        初始化Vision Transformer（双分支架构）
        
        Args:
            img_size: 图像大小
            patch_size: patch大小
            in_channels: 输入通道数
            num_classes: 类别数
            embed_dim: 嵌入维度
            depth: Transformer深度（用于分支2的ViT encoder）
            num_heads: 注意力头数
            mlp_dim: MLP维度
            dropout: Dropout率
            num_topics: 主题数量（用于监督主题模型）
            use_pgcl: 是否使用物理可引导卷积层
            use_dual_branch: 是否使用双分支架构
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.use_dual_branch = use_dual_branch
        self.img_size = img_size
        self.in_channels = in_channels
        
        # 延迟导入避免循环导入
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        from PhysicsGuidedCouplingLayer import PhysicsGuidedCouplingLayer
        
        self.pgcl_hidden_dim = 128
        
        if use_dual_branch and use_pgcl:
            # ========== 分支2：RSD数据通过ViT encoder提取高阶特征 ==========
            self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
            num_patches = self.patch_embed.num_patches

            # 分类 token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # 位置编码
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=dropout)

            # Transformer 编码层（三个堆叠的ViT encoder）
            self.blocks = nn.ModuleList([
                TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            
            # ========== 双分支融合PGCL层 ==========
            # 导入融合PGCL层
            from PhysicsGuidedCouplingLayer import DualBranchFusionPGCL
            
            # 创建融合PGCL层（包含分支1和分支2的处理）
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
            
            # L_CE^e head is DualBranchFusionPGCL.e_classifier (see pgcl_classifier property)
            
        elif use_pgcl:
            # 单分支模式（原始设计，保持向后兼容）
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
            # 不使用PGCL的原始模式
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

        # 参数初始化
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'head'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x):
        B = x.size(0)
        
        # 如果使用双分支架构
        if self.use_dual_branch and self.use_pgcl:
            # ========== 分支2：RSD数据通过ViT encoder提取高阶特征 ==========
            # 通过ViT encoder提取特征
            x_vit = self.patch_embed(x)  # (B, num_patches, embed_dim)
            
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x_vit = torch.cat((cls_tokens, x_vit), dim=1)  # (B, 1+num_patches, embed_dim)
            x_vit = x_vit + self.pos_embed
            x_vit = self.pos_drop(x_vit)
            
            # 通过三个堆叠的ViT encoder
            for blk in self.blocks:
                x_vit = blk(x_vit)
            
            x_vit = self.norm(x_vit)
            cls_features = x_vit[:, 0]  # 取 [CLS] token (B, embed_dim)
            
            # Shared TopMod/PGCL (Algorithm 2): Branch 2 CLS path for evaluation logits
            outputs = self.fusion_pgcl(cls_features, raw_data=x)
            e2 = outputs[0]
            branch2_topic = outputs[2]
            logits = self.pgcl_classifier(e2)
            return logits, cls_features, branch2_topic
            
        elif self.use_pgcl:
            # 单分支模式（原始设计，保持向后兼容）
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # 取 [CLS] token
            
            topic_representation, _ = self.topic_model(cls_out)  # (B, num_topics)
            enhanced_features = self.pgcl(cls_out, topic_representation)  # (B, hidden_dim)
            logits = self.pgcl_classifier(enhanced_features)
            return logits, cls_out, topic_representation
        else:
            # 不使用PGCL的原始模式
            x = self.patch_embed(x)  # (B, num_patches, embed_dim)

            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)

            x = self.norm(x)
            cls_out = x[:, 0]  # 取 [CLS] token
            
            out = self.head(cls_out)
            return out, cls_out
    
    def extract_features(self, x):
        """
        提取CLS token的特征
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            features: CLS token特征，形状为 (B, embed_dim)
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
        cls_out = x[:, 0]  # 取 [CLS] token
        return cls_out
    
    def get_topic_representation(self, x):
        """
        获取主题表示（用于分析）
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            topic_representation: 主题表示，形状为 (B, num_topics)
        """
        if not self.use_pgcl:
            raise ValueError("模型未启用物理可引导卷积层，无法获取主题表示")
        
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
