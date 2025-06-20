import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformer_predict import predict_sample
import json
import os
import pickle
from sklearn.impute import SimpleImputer

def load_rules(rule_file):
    """加载特征规则"""
    rules = {}
    try:
        # 加载数据文件获取有效特征列表
        try:
            df = pd.read_csv('data/sep.csv', sep='|')
        except:
            df = pd.read_csv('data/sep.csv', sep='|')
        valid_features = df.columns.tolist()[:-1]  # 排除标签列
        
        with open(rule_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    feature, rule = line.split(',')
                    feature = feature.strip()
                    rule = rule.strip()
                    
                    # 验证特征是否存在
                    if feature not in valid_features:
                        print(f"错误: 特征 '{feature}' 不存在于数据文件中")
                        continue
                        
                    # 验证规则格式
                    if len(rule) != 2 or not rule.isdigit():
                        print(f"警告: 规则格式错误 '{rule}'，应为两位数字")
                        continue
                        
                    rules[feature] = {
                        'modifiable': bool(int(rule[0])),
                        'binary': bool(int(rule[1]))
                    }
                except ValueError as e:
                    print(f"警告: 无法解析行 '{line}': {str(e)}")
                    continue
    except FileNotFoundError:
        print(f"错误: 找不到规则文件 '{rule_file}'")
        raise
    except Exception as e:
        print(f"错误: 读取规则文件时发生错误: {str(e)}")
        raise
    
    if not rules:
        raise ValueError("没有找到有效的规则")
        
    print("\n加载的规则:")
    for feature, rule in rules.items():
        print(f"{feature}: 可修改={rule['modifiable']}, 二值={rule['binary']}")
    
    return rules

def load_feature_importance():
    """加载特征重要性信息"""
    try:
        importance_df = pd.read_csv('output/feature_importance_heart.csv')
        # 直接返回特征列表，保持文件中的顺序
        return importance_df['feature'].tolist()
    except Exception as e:
        print(f"加载特征重要性文件失败: {str(e)}")
        raise

def modify_single_feature(data, feature, importance, rule, direction=1):
    """修改单个特征"""
    modified_data = data.copy()
    original_value = modified_data[feature]
    
    # 如果原始值为0，则跳过修改
    if rule['modifiable'] == 0:
        return modified_data, {feature: {
            'original': original_value,
            'modified': original_value,
            'type': 'skipped',
            'reason': 'original value is 0',
            'importance': importance
        }}
    
    if rule['binary']:
        # 如果是二值特征，直接取反
        modified_data[feature] = 1 - modified_data[feature]
        modification_info = {
            'original': original_value,
            'modified': modified_data[feature],
            'type': 'binary',
            'importance': importance
        }
    else:
        # 固定修改比例为5%
        percentage_change = 0.05
        
        # 计算修改后的值（固定5%的变化）
        change = abs(original_value * percentage_change)
        
        # 使用传入的方向
        modified_value = original_value + direction * change
        
        # 确保修改后的值在合理范围内
        if feature in ['age_mean', 'los_icu', 'los_hospital']:
            modified_value = max(0, modified_value)
        elif feature in ['sofa_score', 'sapsii', 'charlson_score_max']:
            modified_value = max(0, min(24, modified_value))
        elif feature in ['chloride_max', 'sodium_max']:
            modified_value = max(0, min(200, modified_value))
        elif feature in ['aniongap_max']:
            modified_value = max(0, min(50, modified_value))
        
        # 格式化修改后的值
        if float(modified_value).is_integer():
            modified_value = int(modified_value)
        else:
            modified_value = round(modified_value, 2)
        
        modified_data[feature] = modified_value
        modification_info = {
            'original': original_value,
            'modified': modified_value,
            'type': 'continuous',
            'direction': direction,
            'percentage_change': percentage_change,
            'importance': importance
        }
    
    return modified_data, {feature: modification_info}

def calculate_euclidean_distance(data1, data2):
    """计算两个数据点之间的欧氏距离"""
    # 确保两个数据都是Series类型
    if isinstance(data1, pd.Series):
        features1 = data1
    else:
        features1 = pd.Series(data1)
    
    if isinstance(data2, pd.Series):
        features2 = data2
    else:
        features2 = pd.Series(data2)
    
    # 计算欧氏距离
    return np.sqrt(np.sum((features1 - features2) ** 2))

def main():
    try:
        # 选择模式
        print("请选择运行模式:")
        print("1: 单样本输入模式")
        print("2: 批量处理50个正样本模式")
        mode = input("请输入选择(1或2): ")
        
        # 加载规则
        rules = load_rules('data/rule01.csv')
        
        # 加载特征重要性顺序
        feature_order = load_feature_importance()
        
        # 加载示例数据以获取特征顺序
        try:
            df = pd.read_csv('data/sep.csv', sep='|')
        except:
            df = pd.read_csv('data/sep.csv', sep='|')
        # 统一标签列处理
        label_col = df.columns[-1]
        df = df.rename(columns={label_col: 'label'})
        print("\n数据样例:")
        print(df.iloc[[0]].drop('label', errors='ignore', axis=1))
        
        if mode == '1':
            # 单样本输入模式
            print("\n请输入要预测的样本数据（用逗号,分隔）:")
            print("特征顺序:")
            for i, col in enumerate(df.drop('label', axis=1).columns):
                print(f"{i+1}. {col}")
            
            try:
                # 获取用户输入
                user_input = input("\n请输入数据（用逗号,分隔，包含label列）: ")
                
                # 处理输入数据，将连续竖线转换为空值
                values = []
                for x in user_input.split(','):
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
                input_data = pd.Series(values, index=df.drop('label', axis=1).columns)
                
                # 处理空值
                print("\n处理空值...")
                imputer = SimpleImputer(strategy='mean')
                input_data = pd.Series(
                    imputer.fit_transform(input_data.values.reshape(1, -1)).flatten(),
                    index=input_data.index
                )
                
                # 进行预测
                prediction, probability = predict_sample(input_data, 'model')
                
                print(f"\n预测结果: {prediction}")
                print(f"预测概率: {probability:.4f}")
                
                # 如果预测为1，则生成反事实样本
                if prediction == 1:
                    print("\n开始生成反事实样本...")
                    max_attempts = 150  # 增加总尝试次数
                    success = False
                    best_distance = float('inf')
                    best_modified_data = None
                    best_features = None
                    best_probability = float('inf')  # 初始化为无穷大
                    all_modified_features = {}  # 记录所有修改过的特征
                    feature_impact = {}  # 记录特征影响值 {特征: {'total': 总变化量, 'count': 修改次数}}
                    
                    # 初始化当前数据和概率
                    current_data = input_data.copy()
                    current_probability = probability
                    
                    # 尝试逐个修改特征
                    feature_index = 0
                    attempts_per_feature = 0
                    total_attempts = 0
                    current_feature = feature_order[feature_index]
                    last_direction = 1  # 初始方向为1
                    
                    while total_attempts < max_attempts and feature_index < len(feature_order):
                        total_attempts += 1
                        attempts_per_feature += 1
                        current_feature = feature_order[feature_index]
                        
                        # 从当前最佳数据开始修改
                        if best_modified_data is not None:
                            current_data = best_modified_data.copy()
                            current_probability = best_probability
                        
                        # 修改当前特征
                        modified_data, modified_features = modify_single_feature(
                            current_data, 
                            current_feature, 
                            1.0,  # 使用固定重要性值
                            rules[current_feature],
                            last_direction
                        )
                        
                        # 预测修改后的样本
                        counter_prediction, counter_probability = predict_sample(modified_data, 'model')
                        
                        # 计算欧氏距离
                        distance = calculate_euclidean_distance(input_data, modified_data)
                        
                        # 输出调试信息
                        if total_attempts % 10 == 0 or feature_index < 5:
                            print(f"\n=== 第{total_attempts}次尝试 (特征: {current_feature}) ===")
                            print(f"修改: {modified_features[current_feature]['original']:.4f} -> {modified_features[current_feature]['modified']:.4f}")
                            print(f"预测概率: {counter_probability:.4f} (之前: {current_probability:.4f})")
                            print(f"欧氏距离: {distance:.4f}")
                        
                        # 如果预测概率小于0.45，直接保存结果并停止
                        if counter_probability < 0.45:
                            if counter_probability < best_probability:
                                best_distance = distance
                                best_modified_data = modified_data.copy()
                                best_features = modified_features
                                best_probability = counter_probability
                                
                                # 记录有效概率变化
                                delta = current_probability - counter_probability
                                if current_feature not in feature_impact:
                                    feature_impact[current_feature] = {'total': 0.0, 'count': 0}
                                feature_impact[current_feature]['total'] += delta
                                feature_impact[current_feature]['count'] += 1
                                success = True
                            break  # 直接停止修改
                        
                        # 判断当前修改效果
                        probability_change = abs(counter_probability - current_probability)
                        if probability_change < 0.0005:
                            # 概率变化太小，放弃这次修改，转向下一个特征
                            print(f"\n特征 {current_feature} 修改效果不明显，转向下一个特征")
                            feature_index += 1
                            attempts_per_feature = 0
                            last_direction = 1
                            if feature_index < len(feature_order):
                                current_feature = feature_order[feature_index]
                                print(f"开始修改特征: {current_feature}")
                            continue
                        
                        if counter_probability < current_probability:
                            # 修改有效，降低了概率，保持方向继续修改
                            # 记录修改
                            if current_feature not in all_modified_features:
                                all_modified_features[current_feature] = []
                            all_modified_features[current_feature].append(modified_features[current_feature])
                            
                            # 更新最佳数据（基于预测概率）
                            if counter_probability < best_probability:
                                best_distance = distance
                                best_modified_data = modified_data.copy()
                                best_features = modified_features
                                best_probability = counter_probability
                                
                                # 记录有效概率变化
                                delta = current_probability - counter_probability
                                if current_feature not in feature_impact:
                                    feature_impact[current_feature] = {'total': 0.0, 'count': 0}
                                feature_impact[current_feature]['total'] += delta
                                feature_impact[current_feature]['count'] += 1
                        else:
                            # 修改无效，增加了概率，改变方向
                            last_direction = -last_direction
                        
                        # 如果当前特征已尝试20次，转向下一个特征
                        if attempts_per_feature >= 6:
                            print(f"\n特征 {current_feature} 已尝试12次，转向下一个特征")
                            feature_index += 1
                            attempts_per_feature = 0
                            last_direction = 1
                            if feature_index < len(feature_order):
                                current_feature = feature_order[feature_index]
                                print(f"开始修改特征: {current_feature}")
                    
                    if success:
                        print("\n成功生成反事实数据:")
                        print(f"欧氏距离: {best_distance:.4f}")
                        print(f"预测概率: {best_probability:.4f}")
                        print("\n所有修改过的特征:")
                        for feature, modifications in all_modified_features.items():
                            print(f"\n特征 {feature}:")
                            for mod in modifications:
                                print(f"  修改: {mod['original']:.4f} -> {mod['modified']:.4f}")
                    
                        # 添加特征影响统计
                        print("\n特征平均预测概率影响值:")
                        avg_impact = {}
                        for feat, data in feature_impact.items():
                            if data['count'] > 0:
                                avg = data['total'] / data['count']
                                avg_impact[feat] = avg
                                print(f"  {feat}: 平均影响值 {avg:.4f} (共{data['count']}次有效修改)")
                    
                        # 创建结果DataFrame
                        result_df = pd.DataFrame({
                            'Type': ['Original', 'Counter'],
                            'Prediction': [prediction, counter_prediction],
                            'Probability': [probability, best_probability]
                        })

                        # 将统计结果添加到DataFrame
                        result_df['Avg_Impact'] = ['-', json.dumps(avg_impact)]
                        
                        # 添加所有特征
                        for feature in input_data.index:
                            result_df[feature] = [input_data[feature], best_modified_data[feature]]
                        
                        # 保存结果
                        result_df.to_csv('counterfactual_data.csv', sep=',', index=False)

                    # 更新特征重要性文件
                    if avg_impact:
                        importance_path = 'output/feature_importance_heart.csv'
                        if os.path.exists(importance_path) and avg_impact:
                    # 仅在成功生成时更新
                            original_df = pd.read_csv(importance_path)
                            impact_df = pd.DataFrame.from_dict(avg_impact, orient='index', columns=['avg_impact'])
                            merged_df = original_df.merge(impact_df, left_on='feature', right_index=True, how='left')
                            merged_df['sort_key'] = merged_df['avg_impact'].fillna(-np.inf)
                            sorted_df = merged_df.sort_values(by=['sort_key', 'importance'], ascending=[False, False])
                            sorted_df[['feature', 'importance']].to_csv(importance_path, index=False)
                            print(f"特征重要性文件已更新，新顺序前三位: {sorted_df['feature'].head(3).tolist()}")
                        
                        # 新增动态排序特征逻辑
                        importance_path = 'output/feature_importance_heart.csv'
                        if os.path.exists(importance_path) and avg_impact:
                            original_df = pd.read_csv(importance_path)
                            
                            # 创建影响值DataFrame并合并
                            impact_df = pd.DataFrame.from_dict(avg_impact, orient='index', columns=['avg_impact'])
                            merged_df = original_df.merge(impact_df, left_on='feature', right_index=True, how='left')
                            
                            # 根据影响值排序（降序），未修改特征保持原顺序
                            merged_df['sort_key'] = merged_df['avg_impact'].fillna(-np.inf)
                            sorted_df = merged_df.sort_values(by=['sort_key', 'importance'], ascending=[False, False])
                            
                            # 保留原始importance值并保存
                            sorted_df[['feature', 'importance']].to_csv(importance_path, index=False)
                            print(f"\n特征重要性文件已更新，新顺序前三位: {sorted_df['feature'].head(3).tolist()}")
                    else:
                        print("\n无法生成有效的反事实样本")
                        print("建议：")
                        print("1. 尝试增加修改的特征数量")
                        print("2. 尝试增加修改的幅度")
                        print("3. 检查特征规则文件是否正确")
                else:
                    print("\n输入样本预测为0，不需要生成反事实样本")
                    
            except ValueError as e:
                print(f"\n输入格式错误: {str(e)}")
                print("请确保输入的是用逗号,分隔的数值，连续竖线表示空值")
            except Exception as e:
                print(f"\n处理过程中出错: {str(e)}")
        elif mode == '2':
            # 批量处理50个正样本模式
            print("\n开始批量处理前50个标签为1的正样本...")
            # 加载并筛选数据
            try:
                df = pd.read_csv('data/sep.csv', sep='|')
            except:
                df = pd.read_csv('data/sep.csv', sep='|')
            # 统一标签列处理
            label_col = df.columns[-1]
            df = df.rename(columns={label_col: 'label'})
            # 筛选标签为1的前50个样本
            positive_samples = df[df['label'] == 1].head(50)
            print(f"\n已筛选出{len(positive_samples)}个标签为1的样本，开始处理...")
            
            # 初始化结果DataFrame
            all_results = pd.DataFrame()
            
            for idx, sample in positive_samples.iterrows():
                avg_impact = {}  # 初始化空字典
                print(f"\n=== 处理第 {idx+1} 个样本 ===")
                
                # 创建输入数据
                input_data = sample.drop('label')
                
                # 进行预测
                prediction, probability = predict_sample(input_data, 'model')
                
                print(f"预测结果: {prediction}")
                print(f"预测概率: {probability:.4f}")
                
                # 如果预测为1，则生成反事实样本
                if prediction == 1:
                    print("开始生成反事实样本...")
                    max_attempts = 150
                    success = False
                    best_distance = float('inf')
                    best_modified_data = None
                    best_features = None
                    best_probability = float('inf')
                    all_modified_features = {}
                    feature_impact = {}  # 初始化特征影响字典
                    
                    # 初始化当前数据和概率
                    current_data = input_data.copy()
                    current_probability = probability
                    
                    # 尝试逐个修改特征
                    feature_index = 0
                    attempts_per_feature = 0
                    total_attempts = 0
                    current_feature = feature_order[feature_index]
                    last_direction = 1
                    feature_impact = {}
                    
                    while total_attempts < max_attempts and feature_index < len(feature_order):
                        total_attempts += 1
                        attempts_per_feature += 1
                        current_feature = feature_order[feature_index]
                        
                        if best_modified_data is not None:
                            current_data = best_modified_data.copy()
                            current_probability = best_probability
                        
                        modified_data, modified_features = modify_single_feature(
                            current_data, 
                            current_feature, 
                            1.0, 
                            rules[current_feature],
                            last_direction
                        )
                        
                        counter_prediction, counter_probability = predict_sample(modified_data, 'model')
                        distance = calculate_euclidean_distance(input_data, modified_data)
                        
                        if counter_probability < 0.45:
                            if counter_probability < best_probability:
                                best_distance = distance
                                best_modified_data = modified_data.copy()
                                best_features = modified_features
                                best_probability = counter_probability
                                
                                # 记录有效概率变化
                                delta = current_probability - counter_probability
                                if current_feature not in feature_impact:
                                    feature_impact[current_feature] = {'total': 0.0, 'count': 0}
                                feature_impact[current_feature]['total'] += delta
                                feature_impact[current_feature]['count'] += 1
                                success = True
                            break
                        
                        probability_change = abs(counter_probability - current_probability)
                        if probability_change < 0.0005:
                            feature_index += 1
                            attempts_per_feature = 0
                            last_direction = 1
                            if feature_index < len(feature_order):
                                current_feature = feature_order[feature_index]
                            continue
                        
                        if counter_probability < current_probability:
                            if current_feature not in all_modified_features:
                                all_modified_features[current_feature] = []
                            all_modified_features[current_feature].append(modified_features[current_feature])
                            
                            if counter_probability < best_probability:
                                best_distance = distance
                                best_modified_data = modified_data.copy()
                                best_features = modified_features
                                best_probability = counter_probability
                                
                                # 记录有效概率变化
                                delta = current_probability - counter_probability
                                if current_feature not in feature_impact:
                                    feature_impact[current_feature] = {'total': 0.0, 'count': 0}
                                feature_impact[current_feature]['total'] += delta
                                feature_impact[current_feature]['count'] += 1
                        else:
                            last_direction = -last_direction
                        
                        if attempts_per_feature >=6:
                            feature_index += 1
                            attempts_per_feature = 0
                            last_direction = 1
                            if feature_index < len(feature_order):
                                current_feature = feature_order[feature_index]
                    
                    if success:
                        print(f"成功生成反事实数据，概率: {best_probability:.4f}")
                        
                        # 创建当前样本的结果DataFrame
                        result_df = pd.DataFrame({
                            'Type': ['Original', 'Counter'],
                            'Prediction': [prediction, counter_prediction],
                            'Probability': [probability, best_probability]
                        })
                        
                        # 添加所有特征
                        for feature in input_data.index:
                            result_df[feature] = [input_data[feature], best_modified_data[feature]]
                        
                        # 添加到总结果
                        all_results = pd.concat([all_results, result_df], ignore_index=True)
                        importance_path = 'output/feature_importance_heart.csv'
                        if os.path.exists(importance_path) and avg_impact:
                            original_df = pd.read_csv(importance_path)
                            
                            # 创建影响值DataFrame并合并
                            impact_df = pd.DataFrame.from_dict(avg_impact, orient='index', columns=['avg_impact'])
                            merged_df = original_df.merge(impact_df, left_on='feature', right_index=True, how='left')
                            
                            # 根据影响值排序（降序），未修改特征保持原顺序
                            merged_df['sort_key'] = merged_df['avg_impact'].fillna(-np.inf)
                            sorted_df = merged_df.sort_values(by=['sort_key', 'importance'], ascending=[False, False])
                            
                            # 保留原始importance值并保存
                            sorted_df[['feature', 'importance']].to_csv(importance_path, index=False)
                            print(f"\n特征重要性文件已更新，新顺序前三位: {sorted_df['feature'].head(3).tolist()}")
                        
                    else:
                        print("无法生成有效的反事实样本")
                else:
                    print("样本预测为0，跳过")
            
            # 保存所有结果
            if not all_results.empty:
                all_results.to_csv('batch_counterfactual_data_1_nan.csv', sep=',', index=False)
                print("\n批量处理完成，结果已保存到 batch_counterfactual_dat1.csv")
            else:
                print("\n批量处理完成，但未生成任何有效的反事实样本")
        else:
            print("无效的模式选择，请重新运行程序并输入1或2")

            
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        raise



def update_feature_importance():
    """更新特征重要性文件，交换前两位特征顺序"""
    try:
        df = pd.read_csv('output/feature_importance_heart.csv')
        if len(df) >= 2:
            # 交换前两位特征
            top_features = df.iloc[:2]
            df = pd.concat([top_features[::-1], df.iloc[2:]])
            df.to_csv('output/feature_importance_heart.csv', index=False)
            print('\n当前前两位特征:', df['feature'].values[:2])
            print('特征重要性顺序已更新')
    except Exception as e:
        print(f"更新特征重要性文件时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()