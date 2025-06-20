import pandas as pd
import numpy as np
from transformer_predict import predict_sample
import torch

# 读取规则文件并设置列名（根据实际rule.csv结构调整）
rule_df = pd.read_csv('data/rule01.csv', names=['feature', 'modifiable']).set_index('feature')  # 假设第二列是可修改标志（如10表示可修改）
# 加载并筛选数据

df = pd.read_csv('data/sep.csv', sep='|')
# 统一标签列处理（与训练数据列名保持一致）
label_col = df.columns[-1]
df = df.rename(columns={label_col: 'SepsisLabel'})  # 修正为训练时使用的标签列名
# 筛选标签为1的前50个样本
positive_samples = df[df['SepsisLabel'] == 1].head(50)
# 初始化结果列表
random_data = []

for idx, original in positive_samples.iterrows():
    max_attempts = 100  # 设置最大尝试次数
    attempt = max_attempts
    while attempt > 0:
        attempt -= 1
        # 复制原始数据
        modified = original.copy()
        # 随机修改可改变特征
        for feature in rule_df.index:
            # 根据规则标志判断是否可修改（假设10表示可修改）
            if rule_df.loc[feature, 'modifiable'] == '10':
                # 生成原特征值0.5倍至1.5倍之间的随机值
                lower = original[feature] * 0.5
                upper = original[feature] * 1.5
                modified[feature] = np.random.uniform(lower, upper)
        # 预测概率
        modified =modified.drop('SepsisLabel')
        prediction, prob = predict_sample(modified, 'model')
        if prob < 0.45:
            # 获取原始样本的预测概率（原始样本标签为1）
            original_without_label = original.drop('SepsisLabel')
            original_pred, original_prob = predict_sample(original_without_label, 'model')
            
            # 记录原始数据行（Type=Original）
            random_data.append({
                'Type': 'Original',
                'Prediction': 1,
                'Probability': original_prob,
                'age': original_without_label['age'],
                'sex': original_without_label['sex'],
                'chest pain type': original_without_label['chest pain type'],
                'resting bp s': original_without_label['resting bp s'],
                'cholesterol': original_without_label['cholesterol'],
                'fasting blood sugar': original_without_label['fasting blood sugar'],
                'resting ecg': original_without_label['resting ecg'],
                'max heart rate': original_without_label['max heart rate'],
                'exercise angina': original_without_label['exercise angina'],
                'oldpeak': original_without_label['oldpeak'],
                'ST slope': original_without_label['ST slope']
            })
            
            # 记录反事实数据行（Type=Counter）
            random_data.append({
                'Type': 'Counter',
                'Prediction': 0,
                'Probability': prob,
                'age': modified['age'],
                'sex': modified['sex'],
                'chest pain type': modified['chest pain type'],
                'resting bp s': modified['resting bp s'],
                'cholesterol': modified['cholesterol'],
                'fasting blood sugar': modified['fasting blood sugar'],
                'resting ecg': modified['resting ecg'],
                'max heart rate': modified['max heart rate'],
                'exercise angina': modified['exercise angina'],
                'oldpeak': modified['oldpeak'],
                'ST slope': modified['ST slope']
            })
            break
    else:
        # 尝试次数耗尽仍未生成有效反事实数据
        original_without_label = original.drop('SepsisLabel')
        original_pred, original_prob = predict_sample(original_without_label, 'model')
        # 记录原始数据行（Type=Original）
        random_data.append({
            'Type': 'Original',
            'Prediction': 1,
            'Probability': original_prob,
            'age': original_without_label['age'],
            'sex': original_without_label['sex'],
            'chest pain type': original_without_label['chest pain type'],
            'resting bp s': original_without_label['resting bp s'],
            'cholesterol': original_without_label['cholesterol'],
            'fasting blood sugar': original_without_label['fasting blood sugar'],
            'resting ecg': original_without_label['resting ecg'],
            'max heart rate': original_without_label['max heart rate'],
            'exercise angina': original_without_label['exercise angina'],
            'oldpeak': original_without_label['oldpeak'],
            'ST slope': original_without_label['ST slope']
        })
        # 记录失败的反事实数据行（Type=FailedGeneration）
        random_data.append({
            'Type': 'FailedGeneration',
            'Prediction': None,
            'Probability': None,
            'age': None,
            'sex': None,
            'chest pain type': None,
            'resting bp s': None,
            'cholesterol': None,
            'fasting blood sugar': None,
            'resting ecg': None,
            'max heart rate': None,
            'exercise angina': None,
            'oldpeak': None,
            'ST slope': None
        })

# 保存结果（按目标格式调整列顺序）
columns = ['Type', 'Prediction', 'Probability', 'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']
pd.DataFrame(random_data, columns=columns).to_csv('random_data.csv', index=False)