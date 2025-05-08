from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
import jwt
import datetime
import subprocess
import json
import glob
import threading
import shutil
import numpy as np
import torch
from predict import get_args, main as predict_main
from werkzeug.security import generate_password_hash, check_password_hash
import csv

app = Flask(__name__)
# 配置CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# 移除重复的CORS头部
@app.after_request
def after_request(response):
    # 移除重复的头部
    if 'Access-Control-Allow-Origin' in response.headers:
        del response.headers['Access-Control-Allow-Origin']
    if 'Access-Control-Allow-Headers' in response.headers:
        del response.headers['Access-Control-Allow-Headers']
    if 'Access-Control-Allow-Methods' in response.headers:
        del response.headers['Access-Control-Allow-Methods']
    if 'Access-Control-Allow-Credentials' in response.headers:
        del response.headers['Access-Control-Allow-Credentials']
    return response

# 处理OPTIONS请求
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

# 存储预测任务进度
prediction_progress = {}

# 确保必要的目录存在
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ETT')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 配置SQLite数据库
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_forecast.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
db = SQLAlchemy(app)

def init_db_once():
    """仅在数据库文件不存在时初始化表结构，不删除已有数据"""
    db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_forecast.db')
    if not os.path.exists(db_file):
        print(f"数据库文件不存在，自动初始化表结构: {db_file}")
        with app.app_context():
            db.create_all()
            print("Created new tables")

# 用户模型
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    datasets = db.relationship('Dataset', backref='user', lazy=True, cascade='all, delete-orphan')

# 预测结果模型（先定义因为被Dataset引用）
class Prediction(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id', ondelete='CASCADE'), nullable=False)
    prediction_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    result = db.Column(db.Text, nullable=False)  # 存储预测结果
    parameters = db.Column(db.Text, nullable=False)  # 存储预测参数

# 数据集模型
class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    data = db.Column(db.Text, nullable=False)  # 存储CSV数据
    predictions = db.relationship('Prediction', backref='dataset', lazy=True, cascade='all, delete-orphan')

# 密钥，用于JWT
SECRET_KEY = 'your-secret-key'

# 初始化数据库（只在数据库文件不存在时自动初始化）
init_db_once()

def run_prediction_task(dataset_path, model_name, prediction_id):
    """在后台运行预测任务"""
    # 创建应用上下文
    with app.app_context():
        try:
            prediction = Prediction.query.get(prediction_id)
            if not prediction:
                return
                
            prediction_progress[prediction_id] = {
                'status': 'running',
                'progress': 0,
                'message': '开始预测...'
            }

            # 确保数据文件在正确的位置
            data_filename = 'ETTh1.csv'
            target_path = os.path.join('data/ETT', data_filename)
            os.makedirs('data/ETT', exist_ok=True)
            
            # 复制数据文件到目标位置
            shutil.copy2(dataset_path, target_path)
            
            # 更新进度
            prediction_progress[prediction_id].update({
                'progress': 10,
                'message': '数据准备完成，开始预测...'
            })

            # 设置预测参数
            args = get_args()
            args.root_path = './data/ETT/'
            args.data_path = data_filename
            args.data = 'ETTh1'
            args.model = 'TimesNet'
            args.features = 'M'
            args.seq_len = 96
            args.label_len = 48
            args.pred_len = 6
            args.batch_size = 32
            args.model_id = 'TimesNet'

            # 设置模型权重路径为TimeXer6.pth
            model_path = r'E:/学习/基于大模型的电力负荷预测方法研究与开发/系统/临时版/时间序列集合/checkpoints/TimeXer6.pth'
            print(f"使用模型文件: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            args.checkpoints = model_path

            # 更新进度
            prediction_progress[prediction_id].update({
                'progress': 20,
                'message': '模型加载完成，开始预测...'
            })

            # 定义进度回调
            def progress_callback(percent, msg):
                prediction_progress[prediction_id].update({
                    'progress': percent,
                    'message': msg
                })
            # 运行预测
            metrics = predict_main(args, progress_callback=progress_callback)

            # 转换metrics为float类型
            if metrics:
                for k in metrics:
                    if isinstance(metrics[k], (np.floating, np.float32, np.float64)):
                        metrics[k] = float(metrics[k])

            # 创建数据集特定的结果目录
            dataset_results_dir = os.path.join(RESULTS_DIR, str(prediction.dataset_id))
            os.makedirs(dataset_results_dir, exist_ok=True)

            # 动态获取数据集名，拼接图片目录
            dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
            source_dir = f'test_results/{dataset_name}_long_term_forecast_ETTh1_6h_TimesNet_ETTh1_ftM_sl96_ll48_pl6_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
            print(f"查找源目录: {source_dir}")
            
            moved_files = []  # 保证无论是否进入if分支都已定义
            if os.path.exists(source_dir):
                print(f"源目录存在，开始处理预测结果...")
                
                # 调用analyze_predictions.py处理预测结果
                try:
                    # 准备输入文件路径（使用绝对路径）
                    pred_path = os.path.abspath(os.path.join(source_dir, 'prediction.npy'))
                    true_path = os.path.abspath(os.path.join(source_dir, 'groundtruth.npy'))
                    
                    print(f"预测文件路径: {pred_path}")
                    print(f"真实值文件路径: {true_path}")
                    
                    if os.path.exists(pred_path) and os.path.exists(true_path):
                        # 生成对比图和分析结果
                        print("开始生成预测对比图...")
                        from analyze_predictions import analyze_and_visualize
                        analysis_results = analyze_and_visualize(pred_path, true_path, source_dir)
                        print("预测对比图生成完成")
                        print(f"分析结果: {analysis_results}")
                        
                        # 将所有numpy类型转换为Python原生类型
                        converted_results = {}
                        for key, value in analysis_results.items():
                            if isinstance(value, (np.floating, np.float32, np.float64)):
                                converted_results[key] = float(value)
                            else:
                                converted_results[key] = value
                        metrics.update(converted_results)
                        
                        # 移动生成的预测结果文件
                        result_files = {
                            'prediction_vs_groundtruth.png': '预测对比图',
                            'prediction_vs_groundtruth.csv': '预测结果数据'
                        }
                        
                        moved_files = []
                        for filename, file_desc in result_files.items():
                            source_path = os.path.join(source_dir, filename)
                            if os.path.exists(source_path):
                                print(f"找到{file_desc}: {source_path}")
                                
                                # 确保目标目录存在
                                os.makedirs(dataset_results_dir, exist_ok=True)
                                print(f"确保目标目录存在: {dataset_results_dir}")
                                
                                # 生成带时间戳的新文件名
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                new_filename = f"{timestamp}_{filename}"
                                target_path = os.path.join(dataset_results_dir, new_filename)
                                
                                try:
                                    print(f"复制文件: {source_path} -> {target_path}")
                                    shutil.copy2(source_path, target_path)
                                    moved_files.append(new_filename)
                                    print(f"{file_desc}复制成功")
                                    
                                    # 验证目标文件是否存在
                                    if os.path.exists(target_path):
                                        print(f"确认：目标文件已成功创建: {target_path}")
                                    else:
                                        print(f"错误：目标文件未能创建: {target_path}")
                                except Exception as e:
                                    print(f"复制{file_desc}时出错: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"警告：{file_desc}不存在: {source_path}")

                        # 更新预测记录
                        try:
                            prediction_result = {
                                'status': 'completed',
                                'metrics': metrics,
                                'message': '预测完成',
                                'image_files': moved_files
                            }
                            prediction.result = json.dumps(prediction_result)
                            db.session.commit()
                            print(f"预测记录已更新，移动的文件列表: {moved_files}")

                            # 更新进度
                            prediction_progress[prediction_id].update({
                                'status': 'completed',
                                'progress': 100,
                                'message': '预测完成',
                                'metrics': metrics,
                                'image_files': moved_files
                            })
                        except Exception as e:
                            print(f"更新预测记录时出错: {str(e)}")
                            traceback.print_exc()
                    else:
                        print(f"警告：对比图文件不存在: {source_path}")
                        print("列出源目录中的文件:")
                        for file in os.listdir(source_dir):
                            print(f"- {file}")
                except Exception as e:
                    print(f"生成预测对比图时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"源目录不存在: {source_dir}")

            # === 新增：保存指标到 CSV ===
            metrics_csv_path = os.path.join(RESULTS_DIR, 'metrics.csv')
            metrics_row = {
                'prediction_id': prediction.id,
                'dataset_id': prediction.dataset_id,
                'mae': metrics.get('mae'),
                'mse': metrics.get('mse'),
                'rmse': metrics.get('rmse'),
                'mape': metrics.get('mape'),
                'mspe': metrics.get('mspe'),
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            write_header = not os.path.exists(metrics_csv_path)
            with open(metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics_row)
            # === 新增结束 ===

        except Exception as e:
            print(f"预测任务出错: {str(e)}")
            prediction_progress[prediction_id] = {
                'status': 'error',
                'progress': 0,
                'message': f'预测失败: {str(e)}'
            }
            if prediction:
                prediction.result = json.dumps({
                    'status': 'error',
                    'error': str(e)
                })
                db.session.commit()

        finally:
            # 清理临时文件
            try:
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                if os.path.exists(target_path):
                    os.remove(target_path)
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")

def validate_dataset(df):
    """验证数据集格式是否正确"""
    required_columns = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"数据集缺少必需的列: {', '.join(missing_columns)}")
    
    # 验证日期格式
    try:
        pd.to_datetime(df['date'])
    except:
        raise ValueError("date列的格式不正确，应为yyyy-MM-dd HH:mm:ss格式")
    
    # 验证数值列
    numeric_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    for col in numeric_columns:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            raise ValueError(f"{col}列包含非数值数据")

def process_dataset(df):
    """预处理数据集"""
    # 确保日期列格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 按时间排序
    df = df.sort_values('date')
    
    # 检查是否有缺失值
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("Warning: 数据集包含缺失值:")
        print(missing_values[missing_values > 0])
    
    return df

# 用户注册
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print("注册请求数据:", data)  # 添加调试信息
        
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': '用户已存在'}), 400
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': '注册成功'}), 201
    except Exception as e:
        print(f"注册时出错: {str(e)}")  # 添加错误日志
        return jsonify({'error': f'注册失败: {str(e)}'}), 500

# 用户登录
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'error': '用户名或密码错误'}), 401
    
    token = jwt.encode(
        {'user_id': user.id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        SECRET_KEY,
        algorithm='HS256'
    )
    return jsonify({'token': token}), 200

# 验证token的装饰器
def token_required(f):
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '没有提供token'}), 401
        try:
            data = jwt.decode(token.split(' ')[1], SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except:
            return jsonify({'error': 'token无效或已过期'}), 401
        return f(current_user, *args, **kwargs)
    decorated.__name__ = f.__name__
    return decorated

# 上传数据集
@app.route('/upload', methods=['POST'])
@token_required
def upload_dataset(current_user):
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': '只支持CSV格式的文件'}), 400
    
    try:
        # 读取CSV文件内容
        df = pd.read_csv(file)
        
        # 验证数据集格式
        validate_dataset(df)
        
        # 预处理数据集
        df = process_dataset(df)
        
        # 计算基本统计信息
        stats = {
            'total_rows': len(df),
            'time_range': {
                'start': df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'value_ranges': {
                col: {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean())
                } for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
            }
        }
        
        # 保存处理后的数据
        temp_file = os.path.join(UPLOAD_DIR, f'temp_{current_user.id}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        df.to_csv(temp_file, index=False)
        
        # 将DataFrame转换为JSON字符串存储
        dataset = Dataset(
            filename=file.filename,
            user_id=current_user.id,
            data=df.to_json(orient='records', date_format='iso')
        )
        db.session.add(dataset)
        db.session.commit()
        
        return jsonify({
            'message': '数据集上传成功',
            'dataset_id': dataset.id,
            'filename': dataset.filename,
            'stats': stats
        }), 201
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'文件处理错误: {str(e)}'}), 400

# 获取用户的数据集列表
@app.route('/datasets', methods=['GET'])
@token_required
def get_datasets(current_user):
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    return jsonify({
        'datasets': [{
            'id': ds.id,
            'filename': ds.filename,
            'upload_time': ds.upload_time.strftime('%Y-%m-%d %H:%M:%S')
        } for ds in datasets]
    }), 200

# 获取数据集详情
@app.route('/datasets/<int:dataset_id>', methods=['GET'])
@token_required
def get_dataset(current_user, dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id:
        return jsonify({'error': '无权访问此数据集'}), 403
    
    # 将JSON字符串转换回DataFrame
    df = pd.read_json(dataset.data, orient='records')
    df['date'] = pd.to_datetime(df['date'])
    
    # 计算统计信息
    stats = {
        'total_rows': len(df),
        'time_range': {
            'start': df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'end': df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
        },
        'value_ranges': {
            col: {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean())
            } for col in ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        }
    }
    
    return jsonify({
        'filename': dataset.filename,
        'upload_time': dataset.upload_time.strftime('%Y-%m-%d %H:%M:%S'),
        'preview': df.head().to_dict('records'),
        'columns': df.columns.tolist(),
        'row_count': len(df),
        'stats': stats
    }), 200

# 获取可用模型列表
@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    """获取可用的模型列表"""
    try:
        models = [
            {
                'id': 'TimeXer6',
                'name': 'TimeXer 6小时预测',
                'description': '用于6小时电力负荷预测的TimesNet模型',
                'pred_len': 6
            },
            {
                'id': 'TimeXer12',
                'name': 'TimeXer 12小时预测',
                'description': '用于12小时电力负荷预测的TimesNet模型',
                'pred_len': 12
            },
            {
                'id': 'TimeXer24',
                'name': 'TimeXer 24小时预测',
                'description': '用于24小时电力负荷预测的TimesNet模型',
                'pred_len': 24
            }
        ]
        return jsonify({'models': models}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取预测进度
@app.route('/prediction_progress/<int:prediction_id>', methods=['GET', 'OPTIONS'])
def get_prediction_progress(prediction_id):
    if request.method == 'OPTIONS':
        return handle_options('prediction_progress')
        
    try:
        # 验证token
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '没有提供token'}), 401
        try:
            data = jwt.decode(token.split(' ')[1], SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': '用户不存在'}), 401
        except:
            return jsonify({'error': 'token无效或已过期'}), 401

        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user.id:
            return jsonify({'error': '无权访问此预测任务'}), 403
        
        progress = prediction_progress.get(prediction_id, {
            'status': 'unknown',
            'progress': 0,
            'message': '无法获取进度信息'
        })
        
        return jsonify(progress), 200
    except Exception as e:
        print(f"获取预测进度时出错: {str(e)}")
        return jsonify({'error': f'获取预测进度失败: {str(e)}'}), 500

# 获取数据集的预测图片
@app.route('/prediction_images/<int:dataset_id>', methods=['GET'])
@token_required
def get_dataset_images(current_user, dataset_id):
    """获取指定数据集的预测图片"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user.id:
            return jsonify({'error': '无权访问此数据集'}), 403

        # 构建图片目录路径
        dataset_results_dir = os.path.join(RESULTS_DIR, str(dataset_id))
        print(f"查找图片目录: {dataset_results_dir}")
        
        # 如果目录存在，获取所有png图片
        images = []
        if os.path.exists(dataset_results_dir):
            print(f"目录存在，列出所有文件:")
            for file in os.listdir(dataset_results_dir):
                print(f"- {file}")
                if file.endswith('.png'):
                    images.append(file)
            images.sort(reverse=True)  # 最新的图片排在前面
            print(f"找到 {len(images)} 个PNG图片: {images}")
        else:
            print(f"警告：图片目录不存在: {dataset_results_dir}")

        return jsonify({'images': images}), 200
    except Exception as e:
        print(f"获取预测图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 修改获取预测图片的路由
@app.route('/prediction_image/<int:dataset_id>/<path:image_name>', methods=['GET'])
def get_prediction_image(dataset_id, image_name):
    """获取预测结果图片"""
    try:
        # 构建完整的图片路径
        full_path = os.path.join(RESULTS_DIR, str(dataset_id), image_name)
        print(f"请求图片路径: {full_path}")
        
        if os.path.exists(full_path):
            print(f"找到图片文件: {full_path}")
            return send_file(full_path, mimetype='image/png')
        else:
            print(f"警告：图片文件不存在: {full_path}")
            # 列出目录中的文件
            dir_path = os.path.dirname(full_path)
            if os.path.exists(dir_path):
                print(f"目录 {dir_path} 中的文件:")
                for file in os.listdir(dir_path):
                    print(f"- {file}")
            return jsonify({'error': f'Image not found: {image_name}'}), 404
    except Exception as e:
        print(f"获取图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 删除数据集
@app.route('/datasets/<int:dataset_id>', methods=['DELETE'])
@token_required
def delete_dataset(current_user, dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id:
        return jsonify({'error': '无权删除此数据集'}), 403
    
    try:
        # 删除相关的预测结果图片
        task_names = set()
        for pred in dataset.predictions:
            try:
                pred_data = json.loads(pred.result)
                if 'task_name' in pred_data:
                    task_names.add(pred_data['task_name'])
            except:
                pass
        
        for task_name in task_names:
            result_dir = os.path.join('test_results', task_name)
            if os.path.exists(result_dir):
                for file in os.listdir(result_dir):
                    try:
                        os.remove(os.path.join(result_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(result_dir)
                except:
                    pass

        # 删除数据集（会自动删除相关的预测记录）
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({'message': '数据集删除成功'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

# 获取预测历史
@app.route('/predictions', methods=['GET'])
@token_required
def get_predictions(current_user):
    try:
        # 获取用户的所有数据集
        datasets = Dataset.query.filter_by(user_id=current_user.id).all()
        dataset_ids = [ds.id for ds in datasets]
        
        # 获取每个数据集最新的预测记录
        latest_predictions = {}
        for dataset_id in dataset_ids:
            latest_pred = Prediction.query.filter_by(dataset_id=dataset_id)\
                .order_by(Prediction.prediction_time.desc())\
                .first()
            if latest_pred:
                latest_predictions[dataset_id] = latest_pred
        
        # 构建响应数据
        predictions_data = []
        for dataset_id, pred in latest_predictions.items():
            dataset = Dataset.query.get(dataset_id)
            result = json.loads(pred.result) if pred.result else {}
            parameters = json.loads(pred.parameters) if pred.parameters else {}
            
            prediction_data = {
                'id': pred.id,
                'dataset_id': pred.dataset_id,
                'dataset_name': dataset.filename if dataset else 'Unknown',
                'model': parameters.get('model', 'Unknown'),
                'status': result.get('status', 'unknown'),
                'created_at': pred.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                'result': result
            }
            predictions_data.append(prediction_data)
        
        # 按预测时间倒序排序
        predictions_data.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            'predictions': predictions_data
        }), 200
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")  # 调试信息
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取预测历史失败: {str(e)}'}), 500

# 获取单个预测结果
@app.route('/predictions/<int:prediction_id>', methods=['GET', 'OPTIONS'])
def get_prediction(prediction_id):
    if request.method == 'OPTIONS':
        return handle_options('predictions')
        
    try:
        # 验证token
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '没有提供token'}), 401
        try:
            data = jwt.decode(token.split(' ')[1], SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': '用户不存在'}), 401
        except:
            return jsonify({'error': 'token无效或已过期'}), 401

        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user.id:
            return jsonify({'error': '无权访问此预测任务'}), 403
            
        result = json.loads(prediction.result) if prediction.result else {}
        parameters = json.loads(prediction.parameters) if prediction.parameters else {}
        
        return jsonify({
            'id': prediction.id,
            'dataset_id': prediction.dataset_id,
            'dataset_name': dataset.filename if dataset else 'Unknown',
            'model': parameters.get('model', 'Unknown'),
            'status': result.get('status', 'unknown'),
            'created_at': prediction.prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'result': result
        }), 200
    except Exception as e:
        print(f"获取预测结果时出错: {str(e)}")
        return jsonify({'error': f'获取预测结果失败: {str(e)}'}), 500

# 预测
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return handle_options('predict')
        
    try:
        # 验证token
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': '没有提供token'}), 401
        try:
            data = jwt.decode(token.split(' ')[1], SECRET_KEY, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': '用户不存在'}), 401
        except:
            return jsonify({'error': 'token无效或已过期'}), 401

        data = request.get_json()
        print("预测请求数据:", data)
        
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        dataset_id = data.get('dataset_id')
        if not dataset_id:
            return jsonify({'error': '请选择数据集'}), 400
        
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user.id:
            return jsonify({'error': '无权访问此数据集'}), 403
        
        # 将数据集保存为临时文件
        df = pd.read_json(dataset.data)
        temp_file_path = f'temp_{dataset_id}.csv'
        df.to_csv(temp_file_path, index=False)
        
        # 创建预测记录
        model_name = data.get('model_name', 'TimeXer6')  # 默认使用TimeXer6
        pred_len = 6  # 默认预测长度
        
        # 根据选择的模型设置预测长度
        if model_name == 'TimeXer12':
            pred_len = 12
        elif model_name == 'TimeXer24':
            pred_len = 24
            
        parameters = {
            'model_name': model_name,
            'seq_len': 96,
            'label_len': 48,
            'pred_len': pred_len,
            'train_epochs': 1,
            'batch_size': 32
        }
        
        prediction = Prediction(
            dataset_id=dataset_id,
            result=json.dumps({'status': 'pending'}),
            parameters=json.dumps(parameters)
        )
        db.session.add(prediction)
        db.session.commit()
        
        # 在后台启动预测任务
        thread = threading.Thread(
            target=run_prediction_task,
            args=(temp_file_path, model_name, prediction.id)
        )
        thread.start()
        
        return jsonify({
            'message': '预测任务已启动',
            'prediction_id': prediction.id
        }), 202
        
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

def fix_old_predictions():
    """修复老的 prediction 记录，给缺少 metrics 字段的 result 补充 metrics: null"""
    with app.app_context():
        fixed = 0
        for pred in Prediction.query.all():
            try:
                result = json.loads(pred.result) if pred.result else {}
                if 'metrics' not in result:
                    result['metrics'] = None
                    pred.result = json.dumps(result)
                    fixed += 1
            except Exception as e:
                print(f"修复 prediction id={pred.id} 时出错: {e}")
        db.session.commit()
        print(f"已修复 {fixed} 条 prediction 记录")

# 删除预测结果
@app.route('/predictions/<int:prediction_id>', methods=['DELETE'])
@token_required
def delete_prediction(current_user, prediction_id):
    """删除指定的预测结果"""
    try:
        # 获取预测记录
        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user.id:
            return jsonify({'error': '无权删除此预测结果'}), 403
        
        # 删除相关的图片文件
        try:
            result = json.loads(prediction.result) if prediction.result else {}
            image_files = result.get('image_files', [])
            
            if image_files:
                dataset_results_dir = os.path.join(RESULTS_DIR, str(dataset.id))
                print(f"删除预测图片，目录: {dataset_results_dir}")
                
                for image_file in image_files:
                    image_path = os.path.join(dataset_results_dir, image_file)
                    if os.path.exists(image_path):
                        print(f"删除图片: {image_path}")
                        os.remove(image_path)
                    else:
                        print(f"图片不存在: {image_path}")
                
                # 如果目录为空，删除目录
                if os.path.exists(dataset_results_dir) and not os.listdir(dataset_results_dir):
                    print(f"删除空目录: {dataset_results_dir}")
                    os.rmdir(dataset_results_dir)
        except Exception as e:
            print(f"删除图片文件时出错: {str(e)}")
            # 继续删除数据库记录
        
        # 从进度记录中删除
        if prediction_id in prediction_progress:
            del prediction_progress[prediction_id]
        
        # 删除数据库记录
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({
            'message': '预测结果删除成功',
            'prediction_id': prediction_id
        }), 200
        
    except Exception as e:
        print(f"删除预测结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
    fix_old_predictions() 