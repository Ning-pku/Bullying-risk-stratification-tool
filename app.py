from flask import Flask, render_template, request
import joblib
import numpy as np
import pickle
import shap
import warnings
warnings.filterwarnings("ignore", message="Loaded recoded labels:")

app = Flask(__name__)

# 加载模型
with open('scaler_fitted.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
    lightgbm_model_pb = joblib.load('lightgbm_model_Physical Bullying.pkl')
    lightgbm_model_vb = joblib.load('lightgbm_model_Verbal Bullying.pkl')
    lightgbm_model_sb = joblib.load('lightgbm_model_Social Bullying.pkl')
    lightgbm_model_cb = joblib.load('lightgbm_model_Cyber Bullying.pkl')

def load_kmeans_model(dependent_var):
    """加载 KMeans 模型和重新编码的标签"""
    model_data = joblib.load(f'kmeans_model_{dependent_var}.pkl')
    return model_data['kmeans'], model_data['recoding_map']

# 加载 KMeans 模型
kmeans_pb, recoding_map_pb = load_kmeans_model('Physical Bullying')
kmeans_vb, recoding_map_vb = load_kmeans_model('Verbal Bullying')
kmeans_sb, recoding_map_sb = load_kmeans_model('Social Bullying')
kmeans_cb, recoding_map_cb = load_kmeans_model('Cyber Bullying')

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

# 加载 LightGBM 模型
lightgbm_models = {
    'physical': lightgbm_model_pb,
    'verbal': lightgbm_model_vb,
    'social': lightgbm_model_sb,
    'cyber': lightgbm_model_cb
}

# 加载 KMeans 模型和重编码标签
kmeans_models = {
    'physical': (kmeans_pb, recoding_map_pb),
    'verbal': (kmeans_vb, recoding_map_vb),
    'social': (kmeans_sb, recoding_map_sb),
    'cyber': (kmeans_cb, recoding_map_cb)
}

def get_suggestion(risk_level):
    """根据风险级别提供建议"""
    if risk_level == 'High risk':
        return "Immediate intervention recommended. Work with school counselors and parents."
    elif risk_level == 'Middle risk':
        return "Monitor closely and provide support through corresponding educational programs."
    else:
        return "Maintain current approach, no immediate risk."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 假设原始范围如下：
        original_ranges = {
            'Depressive symptoms_High': (0, 30),
            'Family function dissatisfaction': (3, 12),
            'Classmate support_High': (3, 15),
            'Teacher support_High': (3, 15),
            'School belongingness_High': (3, 15),
            'Positive bullying attitude_High': (3, 12)
        }

        # 收集表单数据并进行处理
        data = {
            'Age': int(request.form['Age']),
            'Sex_Male': int(request.form['Sex_Male']),
            'Depressive symptoms_High': (float(request.form['Depressive symptoms_High']) * 30 / 10),  # 恢复原始范围
            'Parent-child relationship_Poor': int(request.form['Parent-child relationship_Poor']),
            'Family function dissatisfaction': (float(request.form['Family function dissatisfaction']) * (12 - 3) / 10) + 3,  # 恢复原始范围
            'Classmate support_High': (float(request.form['Classmate support_High']) * (15 - 3) / 10) + 3,  # 恢复原始范围
            'Teacher support_High': (float(request.form['Teacher support_High']) * (15 - 3) / 10) + 3,  # 恢复原始范围
            'School belongingness_High': (float(request.form['School belongingness_High']) * (15 - 3) / 10) + 3,  # 恢复原始范围
            'School psychosocial course_Yes': int(request.form['School psychosocial course_Yes']),
            'Positive bullying attitude_High': (float(request.form['Positive bullying attitude_High']) * (12 - 3) / 10) + 3,  # 恢复原始范围
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
            selected_indices = [continuous_columns.index(col) if col in continuous_columns else len(continuous_columns) + binary_columns.index(col) for col in selected_columns]

            # 提取对应的输入数据
            model_input_for_type = model_input[:, selected_indices]

            # 进行风险评估
            shap_total_score, risk_level, suggestion, sorted_shap_values = process_bullying_type(bullying_type, model_input_for_type)
            results[bullying_type] = {
                'shap_total_score': shap_total_score,
                'risk_level': risk_level,
                'suggestion': suggestion,
                'shap_values': sorted_shap_values  # 添加 SHAP 值到结果
            }

        return render_template('index.html', result=results)
    
    return render_template('index.html')

def process_bullying_type(bullying_type, model_input_for_type):
    """Process SHAP calculations and risk predictions for each bullying type"""
    lightgbm_model = lightgbm_models[bullying_type]
    explainer = shap.TreeExplainer(lightgbm_model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(model_input_for_type)

    # If shap_values is a list (classification), select the class 1 SHAP values
    if isinstance(shap_values, list):
        if len(shap_values) >= 1:
            # For binary or multi-class classification, select the SHAP values for the positive class
            shap_values = shap_values[1]  # Adjust index if necessary
        else:
            # Handle unexpected case
            raise ValueError("Expected shap_values to have at least one class output.")
    
    # Since model_input_for_type is a single sample, shap_values might have an extra dimension
    if shap_values.ndim > 1:
        # Extract the first (and only) sample's SHAP values
        shap_values = shap_values[0]
    
    columns_needed = bullying_columns_map[bullying_type]

    # Iterate over SHAP values and keep three decimal places
    shap_values_dict = {}
    for i, col in enumerate(columns_needed):
        shap_value = shap_values[i]
        # Ensure shap_value is a scalar
        if isinstance(shap_value, np.ndarray):
            if shap_value.size == 1:
                shap_value = shap_value.item()
            else:
                # Handle unexpected shape
                raise ValueError(f"Expected scalar SHAP value, got array with shape {shap_value.shape}")
        shap_values_dict[col] = round(float(shap_value), 3)

    # Sort SHAP values from largest to smallest
    sorted_shap_values = sorted(
        shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )

    # Compute total SHAP score
    shap_total_score = round(np.sum(shap_values), 3)

    # Use KMeans model for clustering prediction
    kmeans_model, recoding_map = kmeans_models[bullying_type]
    # Ensure input is 2D
    cluster = kmeans_model.predict([shap_values])  # Or use shap_values.reshape(1, -1)
    recoded_label = recoding_map[cluster[0]]
    risk_level = cluster_risk_map[recoded_label]

    # Get suggestion
    suggestion = get_suggestion(risk_level)

    return shap_total_score, risk_level, suggestion, sorted_shap_values


if __name__ == '__main__':
    app.run
