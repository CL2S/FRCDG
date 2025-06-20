import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import seaborn as sns
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import os
import pickle
import torch.nn.functional as F

# Transformer模型相关类
class SepsisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SepsisTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6):  # 增加模型容量
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2)  # 降低dropout率
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.2,  # 降低dropout率
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),  # 增加中间层维度
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

class EnsembleModel:
    def __init__(self, num_models=5, input_dim=10):
        self.models = []
        for _ in range(num_models):
            self.models.append(SepsisTransformer(input_dim=input_dim))
    
    def eval(self):
        """将所有模型设置为评估模式"""
        for model in self.models:
            model.eval()
    
    def predict(self, x, device):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                outputs = model(x.to(device))
                # 添加数值稳定性处理
                outputs = torch.clamp(outputs, min=-50, max=50)
                probs = torch.softmax(outputs, dim=1)
                # 确保概率值有效
                probs = torch.where(torch.isnan(probs), torch.tensor(0.5, device=device), probs)
                predictions.append(probs)
        
        # 平均所有模型的预测概率
        ensemble_probs = torch.stack(predictions).mean(dim=0)
        # 确保最终概率有效
        ensemble_probs = torch.where(torch.isnan(ensemble_probs), torch.tensor(0.5, device=device), ensemble_probs)
        return ensemble_probs

def create_weighted_sampler(labels):
    # 确保标签是整数类型
    labels = labels.astype(np.int64)
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    weights = weights / weights.sum()
    return WeightedRandomSampler(weights, len(weights))

def load_and_balance_data(file_path):
    """加载数据并平衡数据集
    
    处理流程：
    1. 先进行过采样，将正样本从A增加到B
    2. 如果负样本数量大于2B，则随机下采样到2B
    3. 调整类别权重
    """
    print("加载原始数据...")
    # 尝试自动检测分隔符
    try:
        df = pd.read_csv(file_path, sep='|')
    except:
        df = pd.read_csv(file_path, sep='|')
    print(f"原始数据形状: {df.shape}")
    
    # 获取标签列名（最后一列）
    label_col = df.columns[-1]
    print(f"使用列 '{label_col}' 作为标签")
    
    # 确保标签是整数类型
    df[label_col] = df[label_col].astype(np.int64)
    
    # 分离正样本和负样本
    positive_data = df[df[label_col] == 1]
    negative_data = df[df[label_col] == 0]
    
    print(f"\n原始正样本数量: {len(positive_data)}")
    print(f"原始负样本数量: {len(negative_data)}")
    
    # 设置目标正样本数量B
    target_positive_count = 600  # 可以根据需要调整这个值
    
    # 使用SMOTE过采样
    if len(positive_data) < target_positive_count:
        print(f"\n使用SMOTE进行过采样，目标正样本数量: {target_positive_count}")
        X = df.drop(label_col, axis=1)
        y = df[label_col]
        
        # 计算过采样比例
        sampling_ratio = target_positive_count / len(positive_data)
        print(f"过采样比例: {sampling_ratio:.2f}")
        
        try:
            smote = SMOTE(
                random_state=42,
                k_neighbors=min(5, len(positive_data) - 1),
                sampling_strategy={1: target_positive_count},
                n_jobs=-1
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            df = pd.DataFrame(X_resampled, columns=X.columns)
            df[label_col] = y_resampled
            
            print("SMOTE过采样完成")
            
            # 更新正负样本
            positive_data = df[df[label_col] == 1]
            negative_data = df[df[label_col] == 0]
            
            # 对负样本进行下采样，使负样本数量为2B
            target_negative_count = target_positive_count * 2
            
            if len(negative_data) > target_negative_count:
                print(f"\n对负样本进行下采样，从{len(negative_data)}减少到{target_negative_count}")
                negative_data = negative_data.sample(
                    n=target_negative_count,
                    random_state=42
                )
                
                # 合并正负样本
                df = pd.concat([positive_data, negative_data], ignore_index=True)
                print("负样本下采样完成")
            
        except ValueError as e:
            print(f"SMOTE过采样失败: {str(e)}")
            print("使用原始数据集继续处理")
    
    # 打印最终类别分布
    print("\n最终类别分布:")
    print(df[label_col].value_counts(normalize=True))
    
    return df, label_col

def train_ensemble_model(models, train_loader, val_loader, epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu', model_dir='model'):
    """训练集成模型"""
    best_val_acc = 0
    
    for i, model in enumerate(models):
        print(f"\n训练模型 {i+1}/{len(models)}")
        model = model.to(device)
        
        # 计算类别权重，根据正负样本比例1:2调整
        # 负类权重设为0.3，正类权重设为0.7
        class_weights = np.array([0.4, 0.6])
        
        # 创建完整的权重张量
        full_weights = torch.ones(2, device=device)
        for j in range(2):
            if j < len(class_weights):
                full_weights[j] = class_weights[j]
        
        print("\n类别权重:")
        print(full_weights.cpu().numpy())
        
        # 修改Focal Loss的参数
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()

        # 使用修改后的Focal Loss参数
        criterion = FocalLoss(alpha=0.7, gamma=2.0)  # 调整alpha和gamma值
        
        # 修改优化器参数
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=True
        )
        
        best_model_state = None
        best_val_acc_local = 0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_pos_correct = 0
            train_pos_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss):
                    print(f"警告：在第{epoch+1}个epoch出现NaN损失值")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 计算正样本的准确率
                pos_mask = (labels == 1)
                if pos_mask.any():
                    train_pos_total += pos_mask.sum().item()
                    train_pos_correct += ((predicted == labels) & pos_mask).sum().item()
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_pos_correct = 0
            val_pos_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if torch.isnan(loss):
                        print(f"警告：在第{epoch+1}个epoch验证时出现NaN损失值")
                        continue
                        
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # 计算正样本的准确率
                    pos_mask = (labels == 1)
                    if pos_mask.any():
                        val_pos_total += pos_mask.sum().item()
                        val_pos_correct += ((predicted == labels) & pos_mask).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_pos_acc = val_pos_correct / val_pos_total if val_pos_total > 0 else 0
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_pos_acc = train_pos_correct / train_pos_total if train_pos_total > 0 else 0
            
            scheduler.step(val_loss)
            
            # 保存当前模型状态（使用正样本准确率作为指标）
            if val_pos_acc > best_val_acc_local:
                best_val_acc_local = val_pos_acc
                best_model_state = model.state_dict()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Train Acc: {train_acc:.2%}, '
                      f'Train Pos Acc: {train_pos_acc:.2%}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}, '
                      f'Val Acc: {val_acc:.2%}, '
                      f'Val Pos Acc: {val_pos_acc:.2%}')
        
        # 保存最佳模型状态
        model_path = os.path.join(model_dir, f'best_model_{i}.pth')
        print(f"保存模型到: {model_path}")
        torch.save(best_model_state, model_path)
        
        if best_val_acc_local > best_val_acc:
            best_val_acc = best_val_acc_local
    
    return best_val_acc

def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def visualize_positive_probabilities(data_path, model_dir='model', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    ensemble = EnsembleModel(num_models=5, input_dim=10)
    for i in range(5):
        model_path = os.path.join(model_dir, f'best_model_{i}.pth')
        ensemble.models[i].load_state_dict(torch.load(model_path))
    ensemble.eval()

    # 尝试自动检测分隔符
    try:
        df = pd.read_csv(data_path, sep='|')
    except:
        df = pd.read_csv(data_path, sep='|')
    positive_data = df[df['SepsisLabel'] == 1]
    X = positive_data.drop('SepsisLabel', axis=1)
    X_scaled = scaler.transform(X)

    dataset = SepsisDataset(X_scaled, np.zeros(len(X_scaled)))
    dataset.features = dataset.features.to(device)
    dataset.labels = dataset.labels.to(device)
    dataloader = DataLoader(dataset, batch_size=128)
    dataloader = [(x.to(device), y.to(device)) for x, y in dataloader]

    all_probs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            probs = ensemble.predict(inputs, device)
            all_probs.extend(probs[:, 1].cpu().numpy())

    plt.figure(figsize=(10, 6))
    plt.hist(all_probs, bins=50, alpha=0.7, color='#2c7bb6', edgecolor='black')
    plt.title('Positive Samples Prediction Probability Distribution')
    plt.xlabel('Probability of Sepsis')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for fmt in ['pdf', 'png']:
        plt.savefig(os.path.join(output_dir, f'probability_distribution.{fmt}'), 
                   bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close()

    print(f'可视化结果已保存至: {os.path.abspath(output_dir)}')

def evaluate_positive_samples(model, data_path, scaler, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """评估所有正样本的预测结果"""
    # 加载所有数据
    # 尝试自动检测分隔符
    try:
        df = pd.read_csv(data_path, sep='|')
    except:
        df = pd.read_csv(data_path, sep='|')
    positive_data = df[df['SepsisLabel'] == 1]
    print(f"\n正样本总数: {len(positive_data)}")
    
    # 分离特征和标签
    X = positive_data.drop('SepsisLabel', axis=1)
    y = positive_data['SepsisLabel'].values
    
    # 打印特征统计信息
    print("\n特征统计信息:")
    print(X.describe())
    
    # 标准化数据
    X_scaled = scaler.transform(X)
    
    # 创建数据集和数据加载器
    dataset = SepsisDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 预测
    model.eval()
    all_preds = []
    all_probs = []  # 存储预测概率
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 统计预测结果
    pred_ones = sum(all_preds)
    print(f"\n预测为1的数量: {pred_ones}")
    print(f"预测为1的比例: {pred_ones/len(positive_data):.2%}")
    
    # 打印预测概率的统计信息
    probs_array = np.array(all_probs)
    print("\n预测概率统计:")
    print(f"类别0的平均概率: {np.mean(probs_array[:, 0]):.4f}")
    print(f"类别1的平均概率: {np.mean(probs_array[:, 1]):.4f}")
    print(f"最大概率: {np.max(probs_array):.4f}")
    print(f"最小概率: {np.min(probs_array):.4f}")
    
    return np.array(all_preds), y

def plot_results(train_losses, val_losses, y_true, y_pred):
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def predict_single_sample(model, data, scaler, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """预测单个样本"""
    # 标准化数据
    scaled_data = scaler.transform(data.values.reshape(1, -1))
    
    # 创建数据集时直接使用device
    features_tensor = torch.FloatTensor(scaled_data).to(device)
    dataset = SepsisDataset(features_tensor.cpu().numpy(), np.array([0]))
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 确保模型在目标设备
    model = model.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

def train_models(data_path, model_dir, epochs=75):
    """训练模型的主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模型保存目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"创建模型保存目录: {model_dir}")
    
    # 加载数据
    df, label_col = load_and_balance_data(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 分离特征和标签
    X = df.drop(label_col, axis=1)
    y = df[label_col].values
    
    # 处理缺失值
    print("\n处理缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建数据加载器
    train_dataset = SepsisDataset(X_train_scaled, y_train)
    val_dataset = SepsisDataset(X_val_scaled, y_val)
    test_dataset = SepsisDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 打印数据集信息
    print("\n数据集划分信息:")
    print(f"训练集大小: {len(X_train_scaled)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    print("\n训练集类别分布:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\n验证集类别分布:")
    print(pd.Series(y_val).value_counts(normalize=True))
    print("\n测试集类别分布:")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    # 创建和训练集成模型
    input_dim = X_train.shape[1]
    ensemble = EnsembleModel(num_models=5, input_dim=input_dim)
    best_val_acc = train_ensemble_model(ensemble.models, train_loader, val_loader, epochs=epochs, model_dir=model_dir)
    print(f"\n最佳验证准确率: {best_val_acc:.2%}")
    
    # 保存标准化器
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"保存标准化器到: {scaler_path}")
    
    return best_val_acc

def evaluate_models(data_path, model_dir):
    """评估模型的主函数"""
    # 加载标准化器
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 加载数据
    df, label_col = load_and_balance_data(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 分离特征和标签
    X = df.drop(label_col, axis=1)
    y = df[label_col].values
    
    # 处理缺失值
    print("\n处理缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 标准化数据
    X_scaled = scaler.transform(X)
    
    # 创建数据加载器
    dataset = SepsisDataset(X_scaled, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 创建集成模型
    input_dim = X.shape[1]
    ensemble = EnsembleModel(num_models=5, input_dim=input_dim)
    
    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, model in enumerate(ensemble.models):
        model_path = os.path.join(model_dir, f'best_model_{i}.pth')
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    
    # 评估
    print("\n模型评估:")
    ensemble.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    # 修改预测逻辑，使用更低的阈值
    threshold = 0.5  # 将阈值降低到0.5
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            probs = ensemble.predict(inputs, device)
            # 使用更低的阈值进行预测
            predicted = (probs[:, 1] > threshold).long()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 打印评估结果
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))
    print(f"AUC Score: {roc_auc_score(all_labels, all_preds):.4f}")
    
    # 统计预测结果
    pred_ones = sum(all_preds)
    print(f"\n预测为1的数量: {pred_ones}")
    print(f"预测为1的比例: {pred_ones/len(all_labels):.2%}")
    
    # 打印预测概率的统计信息
    probs_array = np.array(all_probs)
    print("\n预测概率统计:")
    print(f"类别0的平均概率: {np.mean(probs_array[:, 0]):.4f}")
    print(f"类别1的平均概率: {np.mean(probs_array[:, 1]):.4f}")
    print(f"最大概率: {np.max(probs_array):.4f}")
    print(f"最小概率: {np.min(probs_array):.4f}")
    
    # 绘制结果
    plot_results([], [], all_labels, all_preds)

def predict_sample(data, model_dir='model'):
    """预测单个样本
    Args:
        data: 可以是DataFrame或Series格式的数据
        model_dir: 模型保存目录
    Returns:
        prediction: 预测结果（0或1）
        probability: 预测概率
    """
    try:
        # 加载标准化器
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # 确保数据是DataFrame格式
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        
        # 获取训练时的特征列表
        feature_names = scaler.feature_names_in_
       
        
        # 确保输入数据的列名与训练时一致
        if not all(col in feature_names for col in data.columns):
            print("警告：输入数据的列名与训练时不一致")
            # 创建新的DataFrame，使用训练时的列名
            new_data = pd.DataFrame(columns=feature_names)
            for col in data.columns:
                if col in feature_names:
                    new_data[col] = data[col]
                else:
                    print(f"警告：特征 '{col}' 在训练数据中不存在")
            data = new_data
        
        # 处理缺失的特征
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            print(f"警告：以下特征缺失，将使用空值填充: {missing_features}")
            for feature in missing_features:
                data[feature] = np.nan
        
        # 确保特征顺序一致
        data = data[feature_names]
        
        # 标准化数据
        scaled_data = scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=feature_names)
        
        # 创建数据集
        dataset = SepsisDataset(scaled_data.values, np.array([0]))  # 标签不重要
        dataloader = DataLoader(dataset, batch_size=1)
        
        # 创建集成模型
        input_dim = len(feature_names)
        ensemble = EnsembleModel(num_models=5, input_dim=input_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 加载所有模型
        for i, model in enumerate(ensemble.models):
            model_path = os.path.join(model_dir, f'best_model_{i}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
        
        # 预测
        ensemble.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                probs = ensemble.predict(inputs, device)
                prediction = (probs[:, 1] > 0.5).long()
                probability = probs[:, 1].cpu().item()
                
                # 确保概率是有效的数值
                if np.isnan(probability):
                    print("警告：预测概率为NaN，使用默认值0.5")
                    probability = 0.5
                
                return prediction.cpu().item(), probability
    
    except Exception as e:
        print(f"预测过程中出错: {str(e)}")
        raise

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data", "sep.csv")
    model_dir = os.path.join(current_dir, "model")
    
    # 选择模式
    mode = input("请选择模式 (1: 训练模型, 2: 评估模型, 4: 可视化正样本概率分布, 5: 退出): ")
    
    if mode == "1":
        print("\n开始训练模型...")
        train_models(data_path, model_dir)
    elif mode == "2":
        print("\n开始评估模型...")
        evaluate_models(data_path, model_dir)
    elif mode == "4":
        print("\n开始可视化正样本概率分布...")
        data_path = input("请输入数据文件路径（默认：data/sep.csv）: ") or "data/sep.csv"
        
        try:
            with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
                scaler = pickle.load(f)
            
            input_dim = len(scaler.feature_names_in_)
            ensemble = EnsembleModel(num_models=5, input_dim=input_dim)
            for i in range(5):
                model_path = os.path.join(model_dir, f'best_model_{i}.pth')
                ensemble.models[i].load_state_dict(torch.load(model_path))
            
            output_dir = input("请输入输出目录（默认：output）: ") or "output"
            visualize_positive_probabilities(data_path, model_dir, output_dir)
        except Exception as e:
            print(f"错误发生：{str(e)}")
    elif mode == "5":
        print("退出程序。")
        return
    else:
        print("无效的选择，请选择1-4")
        
        # 加载示例数据
        # 尝试自动检测分隔符
    try:
        df = pd.read_csv(data_path, sep='|')
    except:
        df = pd.read_csv(data_path, sep='|')
        print("\n数据样例:")
        print(df.iloc[0].drop('SepsisLabel'))
        
        print("\n请输入要预测的样本数据（用竖线|分隔）:")
        print("特征顺序:")
        for i, col in enumerate(df.drop('SepsisLabel', axis=1).columns):
            print(f"{i+1}. {col}")
        
        try:
            # 获取用户输入
            user_input = input("\n请输入数据（用竖线|分隔）: ")
            
            # 处理输入数据，将连续竖线转换为空值
            values = []
            for x in user_input.split('|'):
                x = x.strip()
                if x == '':
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(x))
                    except ValueError:
                        print(f"警告：无法将 '{x}' 转换为数值，将设置为空值")
                        values.append(np.nan)
            
            # 创建输入数据
            input_data = pd.Series(values, index=df.drop('SepsisLabel', axis=1).columns)
            
            # 处理空值
            print("\n处理空值...")
            imputer = SimpleImputer(strategy='mean')
            input_data = pd.Series(
                imputer.fit_transform(input_data.values.reshape(1, -1)).flatten(),
                index=input_data.index
            )
            
            # 进行预测
            prediction, probability = predict_sample(input_data, model_dir)
            
            print(f"\n预测结果: {prediction}")
            print(f"预测概率: {probability:.4f}")
            
            # 显示原始标签（如果有）
            if 'SepsisLabel' in df.columns:
                original_label = df.iloc[0]['SepsisLabel']
                print(f"原始标签: {original_label}")
                
        except ValueError as e:
            print(f"\n输入格式错误: {str(e)}")
            print("请确保输入的是用竖线|分隔的数值，连续竖线表示空值")
        except Exception as e:
            print(f"\n预测过程中出错: {str(e)}")
    

if __name__ == "__main__":
    main()