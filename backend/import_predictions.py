import os
import pandas as pd
import re
from app import app, db
from models import PredictionResult

def import_predictions_from_csv(dataset_name, csv_file_path, feature_index):
    """从CSV文件导入预测数据到数据库"""
    try:
        print(f"正在读取CSV文件: {csv_file_path}")
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        print(f"CSV文件列名: {df.columns.tolist()}")
        
        # 确保CSV文件包含所需的列
        required_columns = ['Time', 'Prediction', 'Groundtruth', 'Absolute_Error']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV文件缺少必要的列: {csv_file_path}")
            print(f"当前列: {df.columns.tolist()}")
            return False
            
        # 将数据导入数据库
        with app.app_context():
            for _, row in df.iterrows():
                prediction = PredictionResult(
                    dataset_name=dataset_name,
                    feature_index=feature_index,
                    time=str(row['Time']),
                    prediction=float(row['Prediction']),
                    actual=float(row['Groundtruth']),
                    error=float(row['Absolute_Error'])
                )
                db.session.add(prediction)
            
            db.session.commit()
            print(f"成功导入数据: {csv_file_path}")
            return True
            
    except Exception as e:
        print(f"导入数据时出错: {str(e)}")
        return False

def import_all_predictions():
    """导入所有预测数据"""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_results')
    print(f"基础目录: {base_dir}")
    
    # 遍历所有数据集目录
    for dataset_dir in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
            
        print(f"处理数据集: {dataset_dir}")
        
        # 遍历所有CSV文件
        for filename in os.listdir(dataset_path):
            if not filename.endswith('.csv') or not 'prediction_vs_groundtruth' in filename:
                continue
                
            # 从文件名中提取特征索引
            feature_match = re.search(r'feature_(\d+)\.csv$', filename)
            if not feature_match:
                continue
                
            feature_index = int(feature_match.group(1))
            csv_path = os.path.join(dataset_path, filename)
            
            print(f"发现CSV文件: {filename}, 特征索引: {feature_index}")
            
            # 导入数据
            import_predictions_from_csv(dataset_dir, csv_path, feature_index)

if __name__ == '__main__':
    # 确保数据库表存在
    with app.app_context():
        db.create_all()
    
    # 导入所有预测数据
    import_all_predictions() 