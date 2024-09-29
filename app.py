from flask import Flask, render_template, request
import joblib
import numpy as np
import pickle
import shap
import warnings
import os

# 忽略特定警告
warnings.filterwarnings("ignore", message="Loaded recoded labels:")

app = Flask(__name__)
current_dir = os.path.dirname(__file__)  # 获取当前文件所在的目录

# 生成标准化器的文件路径
scaler_file_path = os.path.join(current_dir, 'scaler_fitted.pkl')

# 加载标准化器
with open(scaler_file_path, 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# 定义风险级别映射
cluster_risk_map = {0: 'Low risk', 1: 'Middle risk', 2: 'High risk'}

# 每种欺凌类型所需的列映射
bullying_columns_map = {
    'physical': ['Age', 'Sex_Male', 'Depressive symptoms_High', 'Parent-child relationship_Poor', 'Family function dissatisfaction',
                 'Classmate support_High', 'Teacher support_High', 'School belongingness_High',
                 'School psychosocial course_Yes', 'Positive bullying attitude_High'],
    'verbal': ['Age', 'Sex_Male', 'Two week illness_Yes', 'Depressive symptoms_High', 'Parent-child relationship_Poor',
               'Family function dissatisfaction', 'Classmate support_High', 'Teacher support_High',
               'School belongingness_High', 'Positive bullying attitude_High'],
    'social': ['Age', 'Sex_Male', 'Rural', 'Depressive symptoms_High', 'Parent-child relationship_Poor',
               'Family function dissatisfaction', 'Classmate support_High', 'Teacher support_High',
               'School belongingness_High', 'Positive bullying attitude_High'],
    'cyber': ['Age', 'Sex_Male', 'Depressive symptoms_High', 'Parent-child relationship_Poor', 'Family function dissatisfaction',
              'Classmate support_High', 'Teacher support_High', 'School belongingness_High',
              'School psychosocial course_Yes', 'Positive bullying attitude_High']
}

def get_suggestion(risk_level):
    """根据风险级别提供建议"""
    if risk_level == 'High risk':
        return "Immediate intervention recommended. Work with school counselors and parents."
    elif risk_level == 'Middle risk':
        return "Monitor closely and provide support through corresponding educational programs."
    else:
        return "Maintain current approach, no immediate risk."

# 模型和解释器缓存
lightgbm_model_cache = {}
kmeans_model_cache = {}
explainer_cache = {}

def load_lightgbm_model(bullying_type):
    """按需加载 LightGBM 模型"""
    if bullying_type in lightgbm_model_cache:
        return lightgbm_model_cache[bullying_type]
    model_filename = f'lightgbm_model_{bullying_type.title()} Bullying.pkl'
    model_path = os.path.join(current_dir, model_filename)
    model = joblib.load(model_path)
    lightgbm_model_cache[bullying_type] = model
    return model

def load_kmeans_model(bullying_type):
    """按需加载 KMeans 模型和重编码映射"""
    if bullying_type in kmeans_model_cache:
        return kmeans_model_cache[bullying_type]
    model_filename = f'kmeans_model_{bullying_type.title()} Bullying.pkl'
    model_path = os.path.join(current_dir, model_filename)
    model_data = joblib.load(model_path)
    kmeans_model_cache[bullying_type] = (model_data['kmeans'], model_data['recoding_map'])
    return kmeans_model_cache[bullying_type]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 收集表单数据并进行处理
        data = {
            'Age': int(request.form['Age']),
            'Sex_Male': int(request.form['Sex_Male']),
            'Depressive symptoms_High': float(request.form['Depressive symptoms_High']) * 3,  # 恢复原始范围（0-30）
            'Parent-child relationship_Poor': int(request.form['Parent-child relationship_Poor']),
            'Family function dissatisfaction': float(request.form['Family function dissatisfaction']) * 0.9 + 3,  # 恢复原始范围（3-12）
            'Classmate support_High': float(request.form['Classmate support_High']) * 1.2 + 3,  # 恢复原始范围（3-15）
            'Teacher support_High': float(request.form['Teacher support_High']) * 1.2 + 3,     # 恢复原始范围（3-15）
            'School belongingness_High': float(request.form['School belongingness_High']) * 1.2 + 3,  # 恢复原始范围（3-15）
            'School psychosocial course_Yes': int(request.form['School psychosocial course_Yes']),
            'Positive bullying attitude_High': float(request.form['Positive bullying attitude_High']) * 0.9 + 3,  # 恢复原始范围（3-12）
            'Two week illness_Yes': int(request.form['Two week illness_Yes']),
            'Rural': int(request.form['Rural'])
        }

        continuous_columns = ['Age', 'Depressive symptoms_High', 'Parent-child relationship_Poor', 'Family function dissatisfaction',
                              'Classmate support_High', 'Teacher support_High',
                              'School belongingness_High', 'Positive bullying attitude_High']
        binary_columns = ['Sex_Male', 'School psychosocial course_Yes', 'Two week illness_Yes', 'Rural']

        # 将输入数据转换为 numpy 数组
        continuous_input = np.array([data[feature] for feature in continuous_columns]).reshape(1, -1)
        binary_input = np.array([data[feature] for feature in binary_columns]).reshape(1, -1)

        # 标准化连续变量
        standardized_data = loaded_scaler.transform(continuous_input)
        model_input = np.hstack((standardized_data, binary_input))

        # 处理每种欺凌类型
        results = {}
        for bullying_type in ['physical', 'verbal', 'social', 'cyber']:
            # 根据每个欺凌类型提取相应的 model_input 数据
            selected_columns = bullying_columns_map[bullying_type]
            selected_indices = [
                continuous_columns.index(col) if col in continuous_columns else len(continuous_columns) + binary_columns.index(col)
                for col in selected_columns
            ]

            # 提取对应的输入数据
            model_input_for_type = model_input[:, selected_indices]

            # 进行风险评估
            try:
                shap_total_score, risk_level, suggestion, sorted_shap_values = process_bullying_type(bullying_type, model_input_for_type)
                results[bullying_type] = {
                    'shap_total_score': shap_total_score,
                    'risk_level': risk_level,
                    'suggestion': suggestion,
                    'shap_values': sorted_shap_values  # 添加 SHAP 值到结果
                }
            except Exception as e:
                results[bullying_type] = {
                    'error': str(e)
                }

        return render_template('index.html', result=results)
    
    return render_template('index.html')

def process_bullying_type(bullying_type, model_input_for_type):
    """处理每种欺凌类型的 SHAP 计算和风险预测"""
    # 加载 LightGBM 模型
    lightgbm_model = load_lightgbm_model(bullying_type)
    
    # 使用缓存的解释器
    if bullying_type in explainer_cache:
        explainer = explainer_cache[bullying_type]
    else:
        explainer = shap.TreeExplainer(lightgbm_model)
        explainer_cache[bullying_type] = explainer
    
    # 计算 SHAP 值
    shap_values = explainer.shap_values(model_input_for_type)

    # 选择分类模型的正类 SHAP 值
    if isinstance(shap_values, list):
        if len(shap_values) >= 1:
            shap_values = shap_values[1]  # 根据需要调整索引
        else:
            raise ValueError("Expected shap_values to have at least one class output.")
    
    # 提取单个样本的 SHAP 值
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    
    columns_needed = bullying_columns_map[bullying_type]

    # 处理 SHAP 值并保留三位小数
    shap_values_dict = {}
    for i, col in enumerate(columns_needed):
        shap_value = shap_values[i]
        if isinstance(shap_value, np.ndarray):
            if shap_value.size == 1:
                shap_value = shap_value.item()
            else:
                raise ValueError(f"Expected scalar SHAP value, got array with shape {shap_value.shape}")
        shap_values_dict[col] = round(float(shap_value), 3)

    # 按照 SHAP 值从大到小排序
    sorted_shap_values = sorted(
        shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # 计算总的 SHAP 分数
    shap_total_score = round(np.sum(shap_values), 3)

    # 加载 KMeans 模型
    kmeans_model, recoding_map = load_kmeans_model(bullying_type)
    # 确保输入是二维的
    cluster = kmeans_model.predict([shap_values])
    recoded_label = recoding_map[cluster[0]]
    risk_level = cluster_risk_map[recoded_label]

    # 获取建议
    suggestion = get_suggestion(risk_level)

    return shap_total_score, risk_level, suggestion, sorted_shap_values

if __name__ == '__main__':
    app.run()
