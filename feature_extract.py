import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        
        # 构建编码器层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        
        # 构建解码器层
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            self.layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return self.output_layer(z)

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], input_dim)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def compute_factor_importance(model: VAE, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    data.requires_grad_(True)
    
    # 前向传播
    reconstructed = model.decoder(model.encoder(data)[0])
    
    # 创建虚拟梯度
    dummy_grad = torch.ones_like(reconstructed)
    
    # 计算解码器输出对输入的偏导数
    input_grad = torch.autograd.grad(
        outputs=reconstructed,
        inputs=data,
        grad_outputs=dummy_grad,
        create_graph=False,
        retain_graph=True
    )[0]
    
    # 计算特征重要性（梯度的L2范数）
    importance_scores = torch.norm(input_grad, dim=0)
    
    return importance_scores

def train_vae(model: VAE, train_loader: torch.utils.data.DataLoader, 
              num_epochs: int, learning_rate: float = 0.001) -> List[float]:
    """
    训练VAE模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            # 确保batch是张量
            if isinstance(batch, tuple):
                batch = batch[0]
            elif isinstance(batch, list):
                batch = batch[0]
            
            # 确保batch是浮点型张量
            batch = batch.float()
            
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed, mu, log_var = model(batch)
            
            # 计算重构损失
            reconstruction_loss = F.mse_loss(reconstructed, batch, reduction='mean')
            
            # 计算KL散度损失
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            
            # 总损失
            loss = reconstruction_loss + kl_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
    return losses

def load_and_preprocess_data(file_path: str) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    加载并预处理数据
    """
    # 尝试自动检测分隔符
    df = pd.read_csv(file_path, sep='|')    
    # 获取标签列名（最后一列）
    label_col = df.columns[-1]
    print(f"使用列 '{label_col}' 作为标签")
    
    # 删除包含NA的行
    df = df.dropna()
    print("删除NA后的数据形状:", df.shape)
    
    # 分离特征和标签
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    print("特征数据形状:", X.shape)
    
    # 确保所有列都是数值型
    X = X.astype(float)
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("标准化后的形状:", X_scaled.shape)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_scaled)
    print("张量形状:", X_tensor.shape)
    
    return X_tensor, X, label_col

def visualize_importance(importance_scores: torch.Tensor, feature_names: List[str], 
                        top_n: int = 10):
    """
    可视化特征重要性
    """
    # 将重要性得分转换为numpy数组
    scores = importance_scores.numpy()
    
    # 确保top_n不超过特征数量
    top_n = min(top_n, len(feature_names))
    
    # 获取top_n个最重要的特征
    top_indices = np.argsort(scores)[-top_n:]
    top_scores = scores[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # 创建条形图
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_scores)), top_scores)
    plt.yticks(range(len(top_scores)), top_features)
    plt.xlabel('重要性得分')
    plt.title(f'Top {top_n} 重要特征')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def extract_features(df, label_col=None):
    """提取特征"""
    if label_col is None:
        label_col = df.columns[-1]
    X = df.drop(label_col, axis=1)
    return X

# 使用示例
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载数据
    data_path = "data/sep.csv"
    X_tensor, original_df, label_col = load_and_preprocess_data(data_path)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    # 创建模型
    input_dim = X_tensor.shape[1]
    hidden_dims = [64, 32]
    latent_dim = 16
    model = VAE(input_dim, hidden_dims, latent_dim)
    
    # 训练模型
    num_epochs = 100
    losses = train_vae(model, train_loader, num_epochs)
    
    # 计算特征重要性
    importance_scores = compute_factor_importance(model, X_tensor)
    
    # 可视化特征重要性
    feature_names = original_df.columns.tolist()
    visualize_importance(importance_scores, feature_names)
    
    # 打印特征重要性
    print("\n特征重要性得分:")
    for i, (feature, score) in enumerate(zip(feature_names, importance_scores)):
        print(f"{feature}: {score.item():.4f}")
    
    # 打印调试信息
    print(f"\n特征名称数量: {len(feature_names)}")
    print(f"重要性得分数量: {importance_scores.shape[0]}")
    
    # 确保特征名称和重要性得分的长度匹配
    if len(feature_names) != importance_scores.shape[0]:
        print("警告：特征名称和重要性得分的长度不匹配，将截断特征名称")
        feature_names = feature_names[:importance_scores.shape[0]]
    
    # 保存特征重要性到CSV文件
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores.cpu().numpy()
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv('output/feature_importance_heart.csv', index=False)
