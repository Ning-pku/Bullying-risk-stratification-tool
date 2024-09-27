from flask import Flask, render_template, request 
import joblib  # 确保导入 joblib
import numpy as np
import pickle
import shap
import warnings
warnings.filterwarnings("ignore", message="Loaded recoded labels:")

app = Flask(__name__)

# 加载模型
with open('scaler_fitted.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
lightgbm_model = joblib.load('lightgbm_model_physical.pkl')

def load_kmeans_model(dependent_var):
    """加载 KMeans 模型和重新编码的标签"""
    model_data = joblib.load(f'kmeans_model_{dependent_var}.pkl')
    kmeans = model_data['kmeans']  # KMeans 模型
    recoded_labels = model_data['recoded_labels']  # 重新编码的标签
    recoding_map = model_data['recoding_map']  # recoding_map 用于映射原始标签
    return kmeans, recoding_map, recoded_labels

# 加载 KMeans 模型、重新编码的标签和映射
kmeans, recoding_map, recoded_labels = load_kmeans_model('Physical Bullying')
print(f"Loaded recoded labels: {recoded_labels}")

cluster_risk_map = {0: 'Low risk', 1: 'Middle risk', 2: 'High risk'}

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
            'Positive bullying attitude_High': (float(request.form['Positive bullying attitude_High']) * (12 - 3) / 10) + 3  # 恢复原始范围
        }

        continuous_columns = ['Age', 'Depressive symptoms_High', 'Parent-child relationship_Poor', 'Family function dissatisfaction', 
                              'Classmate support_High', 'Teacher support_High', 
                              'School belongingness_High', 'Positive bullying attitude_High']
        binary_columns = ['Sex_Male', 'School psychosocial course_Yes']

        # 将输入数据转换为 numpy 数组
        continuous_input = np.array([data[feature] for feature in continuous_columns]).reshape(1, -1)
        binary_input = np.array([data[feature] for feature in binary_columns]).reshape(1, -1)

        # 标准化连续变量
        standardized_data = loaded_scaler.transform(continuous_input)
        model_input = np.hstack((standardized_data, binary_input))

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(lightgbm_model)
        shap_values = explainer.shap_values(model_input)

        # 计算 SHAP 值的总和
        shap_total_score = np.sum(shap_values)

        # 使用 KMeans 模型进行聚类预测
        cluster = kmeans.predict(shap_values)
        # 使用 recoded_labels 进行重新映射
        recoded_label = recoding_map[cluster[0]]
        risk_level = cluster_risk_map[recoded_label]  # 使用映射后的标签

        result = {'shap_total_score': shap_total_score, 'risk_level': risk_level} #'shap_values': shap_values.tolist() 
        return render_template('index.html', result=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
