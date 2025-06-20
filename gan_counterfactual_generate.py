import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from transformer_predict import predict_sample
from sklearn.impute import SimpleImputer
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes=2):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 8)  # 增加嵌入维度
        self.model = nn.Sequential(
            nn.Linear(input_dim + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, 8)  # 增加嵌入维度
        self.model = nn.Sequential(
            nn.Linear(input_dim + 8, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x)

def load_rules(rule_file):
    rules = {}
    try:
        df = pd.read_csv('data/sep.csv', sep=',')
        valid_features = df.columns.tolist()[:-1]
        with open(rule_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    feature, rule = line.split(',')
                    if feature not in valid_features: continue
                    if len(rule)!=2 or not rule.isdigit(): continue
                    rules[feature] = {
                        'modifiable': bool(int(rule[0])),
                        'binary': bool(int(rule[1]))
                    }
                except:
                    continue
        return rules
    except Exception as e:
        print(f"加载规则文件失败: {str(e)}")
        raise

def enforce_rules(generated_data, original_data, rules):  # 添加original_data参数
    for feature, rule in rules.items():
        if not rule['modifiable']:
            generated_data[feature] = original_data[feature]  # 使用传入的原始数据
        if rule['binary']:
            generated_data[feature] = round(generated_data[feature])
    return generated_data

def train_gan(original_data, epochs=100, lr=0.0002, batch_size=32):
    rules = load_rules('data/rule.csv')
    input_dim = original_data.shape[1]
    if input_dim != len(rules):
        raise ValueError(f'规则文件包含{len(rules)}个特征，与数据维度{input_dim}不匹配')
    # 根据有效特征数量调整输入维度
    # 有效维度应与数据维度一致
    effective_dim = input_dim  # input_dim已通过规则检查，等于规则特征数
    generator = Generator(effective_dim, 64, effective_dim, num_classes=2)
    discriminator = Discriminator(effective_dim, 64, num_classes=2)
    criterion = nn.BCELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i in range(0, len(original_data), batch_size):
            real_data = torch.tensor(original_data[i:i+batch_size].values, dtype=torch.float32)
            current_batch_size = real_data.shape[0]  # 获取当前批次实际大小
            # 创建当前批次的反转标签（原始标签为1，反转后为0）
            reversed_labels = torch.zeros(current_batch_size, dtype=torch.long)
            # 使用当前批次大小创建判别器标签
            real_labels = torch.ones(current_batch_size, dtype=torch.long)
            # 调整噪声维度匹配有效特征和当前批次大小
            z = torch.randn(current_batch_size, effective_dim)
            fake_data = generator(z, reversed_labels)

            # 训练判别器（真实数据带原始标签，生成数据带反转标签）
            dis_optimizer.zero_grad()
            real_labels = torch.ones(current_batch_size, dtype=torch.long)
            # 输出真实数据和生成数据维度
            logging.debug(f'真实数据维度: {real_data.shape}, 生成数据维度: {fake_data.shape}')
            real_pred = discriminator(real_data, real_labels)
            real_loss = criterion(real_pred, torch.ones_like(real_pred))
            fake_pred = discriminator(fake_data.detach(), reversed_labels)
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            dis_loss = (real_loss + fake_loss) / 2
            dis_loss.backward()
            dis_optimizer.step()
            logging.info(f'Epoch {epoch}, Batch {i}: 判别器损失={dis_loss.item():.4f}')

            # 训练生成器（使用反转标签欺骗判别器）
            gen_optimizer.zero_grad()
            fake_pred = discriminator(fake_data, reversed_labels)
            # 特征空间对齐约束
            # 确保生成数据与真实数据维度一致
            generated_data = generator(z, reversed_labels)
            if generated_data.shape[1] != real_data.shape[1]:
                raise ValueError(f'生成数据维度{generated_data.shape[1]}与真实数据维度{real_data.shape[1]}不匹配')
            cos_sim = F.cosine_similarity(generated_data, real_data).mean()
            gen_loss = criterion(fake_pred, torch.ones_like(fake_pred)) + 0.5 * (1 - cos_sim)
            gen_loss.backward()
            gen_optimizer.step()
            logging.info(f'Epoch {epoch}, Batch {i}: 生成器损失={gen_loss.item():.4f}')
    return generator

def generate_counterfactual(original_sample, generator, rules, max_attempts=100):
    original_data = pd.Series(original_sample)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(original_data.values.reshape(1, -1))

    for attempt in range(max_attempts):
        # 使用反转标签生成反事实数据
        reversed_label = torch.zeros(1, dtype=torch.long)
        effective_dim = len(rules)  # 使用规则文件中的特征数量作为有效维度
        z = torch.randn(1, effective_dim)
        fake_data = generator(z, reversed_label).detach().numpy().flatten()
        logging.debug(f'尝试{attempt}生成原始假数据: {fake_data}')
        fake_data = scaler.inverse_transform(fake_data.reshape(1, -1))[0]
        modified_data = pd.Series(fake_data, index=original_data.index)
        modified_data = enforce_rules(modified_data, original_data, rules)
        logging.info(f'尝试{attempt}规则约束后数据: {modified_data.to_dict()}')

        pred, prob = predict_sample(modified_data, 'models')
        logging.info(f'尝试{attempt}预测结果: 类别={pred}, 概率={prob:.4f}')
        if prob < 0.45:
            return modified_data, prob
    return None, None

def main():
    try:
        rules = load_rules('data/rule.csv')

        # 加载sep.csv数据并筛选前50个标签为1的样本
        try:
            df_sep = pd.read_csv('data/sep.csv', sep=',')
            positive_samples = df_sep[df_sep['SepsisLabel'] == 1].head(50)
            if len(positive_samples) < 50:
                raise ValueError(f'数据集中标签为1的样本不足50个，实际有{len(positive_samples)}个')
            df = positive_samples.drop('SepsisLabel', axis=1)
            input_dim = df.shape[1]
            # 输出数据维度和规则特征数
            logging.info(f'数据维度: {input_dim}, 规则特征数: {len(rules)}')
            logging.info(f'数据列名: {df.columns.tolist()}')
        except Exception as e:
            print(f'加载数据失败: {str(e)}')
            return
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        generator = train_gan(pd.DataFrame(scaled_data, columns=df.columns))

        result_df = pd.DataFrame(columns=['Type', 'Prediction', 'Probability'] + df.columns.tolist())
        for idx, row in df.iterrows():
            original_pred, original_prob = predict_sample(row, 'models')
            counter_data, counter_prob = generate_counterfactual(row, generator, rules)
            # 处理反事实生成失败的情况
            original_row = {
                'Type': 'Original',
                'Prediction': original_pred,
                'Probability': original_prob,
                **{col: row[col] for col in df.columns}
            }
            counter_row = {
                'Type': 'Counter' if counter_data is not None else 'Failed',
                'Prediction': 0 if counter_data is not None else 'FailedGeneration',
                'Probability': counter_prob if counter_data is not None else 'FailedGeneration',
                **{col: counter_data[col] if counter_data is not None else 'FailedGeneration' for col in df.columns}
            }
            result_df = pd.concat([result_df, pd.DataFrame([original_row, counter_row])], ignore_index=True)
        result_df.to_csv('gan_counterfactual_data.csv', index=False)
        print("反事实数据生成完成，已保存至gan_counterfactual_data.csv")
    except Exception as e:
        print(f"生成过程出错: {str(e)}")

if __name__ == '__main__':
    main()