import pandas as pd
import numpy as np
import os

def process_missing_values(file_path):
    """
    处理数据文件中的缺失值
    缺失值格式：1|2||3|，其中||之间表示一个缺失值
    填充方法：使用相同标签下该特征的平均值进行填充
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        处理后的DataFrame
    """
    print(f"开始处理文件: {file_path}")
    
    # 读取原始数据
    df = pd.read_csv(file_path, sep='|')
    print(f"原始数据形状: {df.shape}")
    
    # 删除ICULOS列
    if 'ICULOS' in df.columns:
        df = df.drop('ICULOS', axis=1)
        print("已删除ICULOS列")
    
    # 分离特征和标签
    X = df.drop('SepsisLabel', axis=1)
    y = df['SepsisLabel']
    
    # 获取所有特征名
    feature_names = X.columns
    
    # 按标签分组处理缺失值
    for label in y.unique():
        print(f"\n处理标签 {label} 的样本...")
        
        # 获取当前标签的样本索引
        label_mask = y == label
        label_data = X[label_mask]
        
        # 计算当前标签下每个特征的平均值
        feature_means = label_data.mean()
        
        # 对当前标签的样本进行填充
        for feature in feature_names:
            # 获取当前特征的所有值
            feature_values = label_data[feature]
            
            # 找出缺失值的位置
            missing_mask = feature_values.isna()
            
            if missing_mask.any():
                # 使用当前标签下的平均值填充
                X.loc[label_mask & missing_mask, feature] = feature_means[feature]
                print(f"特征 '{feature}' 填充了 {missing_mask.sum()} 个缺失值")
    
    # 重新组合特征和标签
    processed_df = pd.concat([X, y], axis=1)
    
    # 保存处理后的数据
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, 'processed_data.csv')
    processed_df.to_csv(output_file, sep=',', index=False)
    print(f"\n处理后的数据已保存到: {output_file}")
    
    # 打印处理结果统计
    print("\n处理结果统计:")
    print(f"原始样本数: {len(df)}")
    print(f"处理后样本数: {len(processed_df)}")
    print("\n各标签样本数:")
    print(processed_df['SepsisLabel'].value_counts())
    
    return processed_df

def process_data(file_path):
    """
    处理数据文件，包括处理缺失值和数据清洗
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        处理后的DataFrame
    """
    print(f"开始处理文件: {file_path}")
    
    # 读取原始数据
    df = pd.read_csv(file_path, sep='|')
    print(f"原始数据形状: {df.shape}")
    
    # 处理缺失值
    print("\n处理缺失值...")
    df = process_missing_values(file_path)
    
    # 数据清洗
    print("\n数据清洗...")
    # 移除重复行
    df = df.drop_duplicates()
    print(f"移除重复行后的形状: {df.shape}")
    
    # 确保所有数值列都是数值类型
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理异常值
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # 保存处理后的数据
    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, 'cleaned_data.csv')
    df.to_csv(output_file, sep=',', index=False)
    print(f"\n清洗后的数据已保存到: {output_file}")
    
    return df

def main():
    # 设置数据文件路径
    data_path = "data/features/sep.csv"
    
    try:
        # 处理数据
        processed_df = process_data(data_path)
        
        # 打印一些基本统计信息
        print("\n数据基本统计信息:")
        print(processed_df.describe())
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()