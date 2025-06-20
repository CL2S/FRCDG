import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体（修改为Times New Roman）
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  # 替换为目标字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 保持负号显示正常

# 配置参数
CSV_PATH = 'batch_counterfactual_data_1.csv'
OUTPUT_DIR = 'output'
'''
FEATURE_COLS = ['age', 'sex', 'chest pain type', 'resting bp s', 
               'cholesterol', 'fasting blood sugar', 'resting ecg',
               'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']
'''
FEATURE_COLS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取并处理数据
df = pd.read_csv(CSV_PATH)
original = df[df['Type'] == 'Original'][FEATURE_COLS]
counter = df[df['Type'] == 'Counter'][FEATURE_COLS]

# 计算绝对差异均值
diff = (original - counter).abs().mean()

# 创建等宽柱状图
plt.figure(figsize=(24, 12), dpi=120)

# 根据Type类型选择数据对
selected = df[df.index.isin([0,1])]  # 选择第一组数据
original_vals = selected[selected['Type'] == 'Original'][FEATURE_COLS].iloc[0].values
counter_vals = selected[selected['Type'] == 'Counter'][FEATURE_COLS].iloc[0].values

# 标准化并创建比例柱状图
base_height = 1.0
ratios = []
for orig, cnt in zip(original_vals, counter_vals):
    if orig == 0 and cnt == 0:
        ratios.append(1.0)
    else:
        ratios.append(cnt / orig if orig != 0 else 2.0)

x = range(len(FEATURE_COLS))
bar_width = 0.35
bars1 = plt.bar(x, [base_height]*len(x), width=bar_width, label='Original')
bars2 = plt.bar([i + bar_width for i in x], ratios, width=bar_width, label='Counter')

# 设置x轴标签
plt.xticks([i + bar_width/2 for i in x], FEATURE_COLS, rotation=45, ha='right', fontsize=55)
plt.yticks(fontsize=45)

# 设置样式
plt.title('Original vs Counterfactual', fontsize=35, pad=20)
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['legend.fontsize'] = 35
plt.legend()
plt.xticks(rotation=55, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().xaxis.set_ticks_position('bottom')

# 添加带百分比的数值标签
for i, (orig, cnt) in enumerate(zip(original_vals, counter_vals)):
    # 定义格式化函数：>100显示整数，否则保留一位小数
    def format_value(val):
        return f'{val:.0f}' if val > 99 else f'{val:.1f}'
    
    if abs(orig - cnt) < 0.01:  # 浮点精度容错，判断是否相等
        # 相同值时在两柱中间显示一个标签（字体大小25）
        x_pos = i + bar_width/2
        plt.text(x_pos, max(base_height, ratios[i])*1.05, 
                 format_value(orig),  # 应用格式化规则
                 ha='center', va='bottom', fontsize=35, color='purple')
    else:
        # 原始值显示实际数值（蓝色，字体大小25）
        plt.text(i, base_height*1.05, 
                 format_value(orig),  # 应用格式化规则
                 ha='center', va='bottom', fontsize=35, color='blue')
        # 反事实值显示实际值（橙色，字体大小25）
        plt.text(i + bar_width, max(ratios[i], base_height)*1.05, 
                 format_value(cnt),  # 应用格式化规则
                 ha='center', va='bottom', fontsize=35, color='black')

# 隐藏纵轴
plt.gca().get_yaxis().set_visible(False)

plt.tight_layout()
# 循环处理前5组数据
for group_num in range(5):
    start_idx = group_num * 2
    end_idx = start_idx + 1
    
    selected = df[df.index.isin([start_idx, end_idx])]
    original_vals = selected[selected['Type'] == 'Original'][FEATURE_COLS].iloc[0].values
    counter_vals = selected[selected['Type'] == 'Counter'][FEATURE_COLS].iloc[0].values
    
    # 提取预测概率（新增）
    original_prob = selected[selected['Type'] == 'Original']['Probability'].iloc[0]
    counter_prob = selected[selected['Type'] == 'Counter']['Probability'].iloc[0]

    # 重新计算比例
    ratios = []
    for orig, cnt in zip(original_vals, counter_vals):
        if orig == 0 and cnt == 0:
            ratios.append(1.0)
        else:
            ratios.append(cnt / orig if orig != 0 else 2.0)
    
    plt.figure(figsize=(24, 12), dpi=120)
    
    # 设置坐标轴样式
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_ticks_position('bottom')
    
    bars1 = plt.bar(x, [base_height]*len(x), width=bar_width, label='Original')
    bars2 = plt.bar([i + bar_width for i in x], ratios, width=bar_width, label='Counter')
    
    # 设置x轴标签
    plt.xticks([i + bar_width/2 for i in x], FEATURE_COLS, rotation=45, ha='right', fontsize=55)
    plt.yticks(fontsize=45)
    
    # 添加预测概率文本（修改后）
    # 原始概率（蓝色）
    plt.text(0.5 - 0.25, 1.2,  # 调整位置避免重叠
             f'original probability: {original_prob:.2f}',
             ha='center', va='bottom', 
             transform=plt.gca().transAxes,
             fontsize=50, color='blue')  # 蓝色
    
    # 反事实概率（橙色）
    plt.text(0.5 + 0.25, 1.2,  # 调整位置与原始概率对称
             f'counterfactual probability: {counter_prob:.2f}',
             ha='center', va='bottom', 
             transform=plt.gca().transAxes,
             fontsize=50, color='black')  # 橙色
    
    # 添加带百分比的数值标签
    for i, (orig, cnt) in enumerate(zip(original_vals, counter_vals)):
        # 定义格式化函数：>100显示整数，否则保留一位小数
        def format_value(val):
            return f'{val:.0f}' if val > 99 else f'{val:.1f}'
        
        if abs(orig - cnt) < 0.01:  # 浮点精度容错
            x_pos = i + bar_width/2
            plt.text(x_pos, max(base_height, ratios[i])*1.05, 
                     format_value(orig),  # 应用格式化规则
                     ha='center', va='bottom', fontsize=35, color='purple')
        else:
            plt.text(i, base_height*1.05, 
                     format_value(orig),  # 应用格式化规则
                     ha='center', va='bottom', fontsize=35, color='blue')
            plt.text(i + bar_width, max(ratios[i], base_height)*1.05, 
                     format_value(cnt),  # 应用格式化规则
                     ha='center', va='bottom', fontsize=35, color='black')

    # 隐藏纵轴
    plt.gca().get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'counter_{group_num+1}.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
plt.close()