from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token, verify_jwt_in_request
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
import secrets
from datetime import timedelta
from flask_jwt_extended.exceptions import NoAuthorizationError
from backend.models import db, User, Dataset, Prediction, PredictionResult, Comment

app = Flask(__name__)

# CORS配置
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# JWT配置
# JWT_SECRET_KEY = secrets.token_hex(32)
app.config['JWT_SECRET_KEY'] = 'your-very-secret-key-1234567890abcdef'
print(f"Using fixed JWT Secret Key: {app.config['JWT_SECRET_KEY']}")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'
jwt = JWTManager(app)

# 确保必要的目录存在
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ETT')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
UPLOAD_FOLDER = 'uploads'
PREDICTION_FOLDER = 'test_results'

for folder in [UPLOAD_DIR, RESULTS_DIR, UPLOAD_FOLDER, PREDICTION_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 配置SQLite数据库
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_forecast.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}?timeout=30'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_SIZE'] = 1
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 30

# 初始化数据库
db.init_app(app)

# 存储预测任务进度
prediction_progress = {}

# 移除重复的CORS头部
@app.after_request
def after_request(response):
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

# 测试路由
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'})

# 预测数据路由
@app.route('/prediction_data/<dataset>', methods=['GET'])
@jwt_required()
def get_prediction_data(dataset):
    """从数据库获取预测结果数据"""
    try:
        results = PredictionResult.query.filter_by(dataset_name=dataset).order_by(PredictionResult.time).all()
        
        if not results:
            return jsonify({'error': '未找到预测数据'}), 404
            
        data = [result.to_dict() for result in results]
            
        return jsonify({
            'status': 'success',
            'data': data
        })
        
    except Exception as e:
        print(f"获取预测数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 预测图片路由
@app.route('/prediction_image/<dataset>/<filename>', methods=['GET'])
@jwt_required()
def get_prediction_file(dataset, filename):
    """处理预测结果文件的请求（仅图片）"""
    try:
        file_path = os.path.join(PREDICTION_FOLDER, dataset, filename)
        print(f"尝试访问文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return jsonify({'error': f'文件 {filename} 不存在'}), 404
            
        directory = os.path.dirname(file_path)
        base_filename = os.path.basename(file_path)
        
        if filename.endswith('.png'):
            return send_from_directory(
                directory,
                base_filename,
                mimetype='image/png'
            )
        else:
            return jsonify({'error': '不支持的文件类型'}), 400
            
    except Exception as e:
        print(f"处理文件请求时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

def import_prediction_results(dataset_id, source_dir):
    """导入预测结果到数据库"""
    try:
        print(f"开始导入预测结果到数据库，数据集ID: {dataset_id}")
        # 查找所有预测结果CSV文件
        for feature_num in range(3):
            csv_file = glob.glob(os.path.join(source_dir, f'*_prediction_vs_groundtruth_feature_{feature_num}.csv'))
            if not csv_file:
                print(f"找不到特征 {feature_num} 的预测结果文件")
                continue
                
            csv_path = csv_file[0]
            print(f"处理文件: {csv_path}")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                print(f"读取CSV文件成功，列名: {df.columns.tolist()}")
                
                # 删除该数据集之前的预测结果
                PredictionResult.query.filter_by(
                    dataset_name=str(dataset_id),
                    feature_index=feature_num
                ).delete()
                
                # 导入新的预测结果
                for _, row in df.iterrows():
                    result = PredictionResult(
                        dataset_name=str(dataset_id),
                        feature_index=feature_num,
                        time=str(row['Time']),
                        prediction=float(row['Prediction']),
                        actual=float(row['Groundtruth']),
                        error=float(row['Absolute_Error'])
                    )
                    db.session.add(result)
                
                print(f"特征 {feature_num} 的预测结果导入成功")
                
            except Exception as e:
                print(f"处理文件 {csv_path} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 提交所有更改
        db.session.commit()
        print("所有预测结果导入完成")
        return True
        
    except Exception as e:
        print(f"导入预测结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        db.session.rollback()
        return False

def run_prediction_task(dataset_path, model_name, prediction_id):
    """在后台运行预测任务"""
    # 创建应用上下文
    with app.app_context():
        try:
            prediction = Prediction.query.get(prediction_id)
            if not prediction:
                return
                
            prediction_progress[prediction_id] = {
                'status': '预测中...',
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
            args.batch_size = 32
            args.model_id = 'TimesNet'
            args.d_model = 16
            args.n_heads = 8
            args.e_layers = 2
            args.d_ff = 32
            args.dropout = 0.05
            args.enc_in = 3
            args.dec_in = 3
            args.c_out = 3
            args.num_kernels = 6

            # 根据model_name动态设置预测长度和模型权重路径
            if model_name == 'TimeXer6':
                args.pred_len = 6
                model_path = './checkpoints/TimeXer6.pth'
                hour_str = '6h'
            elif model_name == 'TimeXer12':
                args.pred_len = 12
                model_path = './checkpoints/TimeXer12.pth'
                hour_str = '12h'
            elif model_name == 'TimeXer24':
                args.pred_len = 24
                model_path = './checkpoints/TimeXer24.pth'
                hour_str = '24h'
            else:
                # 默认6小时
                args.pred_len = 6
                model_path = './checkpoints/TimeXer6.pth'
                hour_str = '6h'

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
            try:
                metrics = predict_main(args, progress_callback=progress_callback)
                if not metrics:
                    raise Exception("预测失败：没有得到有效的预测结果")
            except Exception as e:
                print(f"预测过程出错: {str(e)}")
                prediction_progress[prediction_id].update({
                    'status': 'error',
                    'progress': 0,
                    'message': f'预测失败: {str(e)}'
                })
                prediction.result = json.dumps({
                    'status': 'error',
                    'error': str(e)
                })
                db.session.commit()
                return

            # 转换metrics为float类型
            if metrics:
                for k in metrics:
                    if isinstance(metrics[k], (np.floating, np.float32, np.float64)):
                        metrics[k] = float(metrics[k])

            # 创建数据集特定的结果目录（每次预测独立子目录）
            sub_dir = f"{model_name}"
            dataset_results_dir = os.path.join(RESULTS_DIR, str(prediction.dataset_id), sub_dir)
            os.makedirs(dataset_results_dir, exist_ok=True)

            # 动态获取数据集名，拼接图片目录（hour_str动态变化）
            dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
            source_dir = f'test_results/{dataset_name}_long_term_forecast_ETTh1_{hour_str}_TimesNet_ETTh1_ftM_sl96_ll48_pl{args.pred_len}_dm16_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
            print(f"查找源目录: {source_dir}")
            
            # 调用 analyze_predictions.py 处理 .npy 文件
            try:
                groundtruth_path = os.path.join(source_dir, 'groundtruth.npy')
                prediction_path = os.path.join(source_dir, 'prediction.npy')
                if os.path.exists(groundtruth_path) and os.path.exists(prediction_path):
                    print("开始处理预测结果...")
                    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_predictions.py')
                    subprocess.run([
                        'python', 
                        analyze_script,
                        '--true_path', groundtruth_path,
                        '--pred_path', prediction_path,
                        '--output_dir', source_dir
                    ], check=True)
                    print("预测结果处理完成")
                else:
                    print(f"警告：找不到预测结果文件")
                    if not os.path.exists(groundtruth_path):
                        print(f"groundtruth.npy 不存在: {groundtruth_path}")
                    if not os.path.exists(prediction_path):
                        print(f"prediction.npy 不存在: {prediction_path}")
            except Exception as e:
                print(f"处理预测结果时出错: {str(e)}")
                import traceback
                traceback.print_exc()

            moved_files = []
            # 查找所有特征的预测结果文件
            for feature_num in range(3):
                result_files = {
                    f'prediction_vs_groundtruth_feature_{feature_num}.png': f'特征{feature_num}预测对比图',
                    f'prediction_vs_groundtruth_feature_{feature_num}.csv': f'特征{feature_num}预测结果数据'
                }
                for filename, file_desc in result_files.items():
                    source_path = os.path.join(source_dir, filename)
                    if os.path.exists(source_path):
                        print(f"找到{file_desc}: {source_path}")
                        # 确保目标目录存在
                        os.makedirs(dataset_results_dir, exist_ok=True)
                        print(f"确保目标目录存在: {dataset_results_dir}")
                        # 生成带时间戳的新文件名
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_filename = f"{timestamp}_{os.path.basename(source_path)}"
                        new_path = os.path.join(dataset_results_dir, new_filename)
                        try:
                            print(f"复制文件: {source_path} -> {new_path}")
                            shutil.copy2(source_path, new_path)
                            moved_files.append(f"{sub_dir}/{new_filename}")
                        except Exception as e:
                            print(f"复制文件失败: {str(e)}")
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
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"预测任务出错: {str(e)}")
            import traceback
            traceback.print_exc()
            prediction_progress[prediction_id] = {
                'status': 'error',
                'progress': 0,
                'message': f'预测失败: {str(e)}'
            }
            if prediction:
                try:
                    prediction.result = json.dumps({
                        'status': 'error',
                        'error': str(e)
                    })
                    db.session.commit()
                except:
                    db.session.rollback()
        finally:
            try:
                # 清理临时文件
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
                if os.path.exists(target_path):
                    os.remove(target_path)
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")
            
            try:
                db.session.close()
            except:
                pass

def validate_dataset(df):
    """验证数据集格式是否正确"""
    required_columns = ['date', 'HUFL', 'MUFL', 'LUFL']  # 使用这三个特征
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"数据集缺少必需的列: {', '.join(missing_columns)}")
    
    # 验证日期格式
    try:
        pd.to_datetime(df['date'])
    except:
        raise ValueError("date列的格式不正确，应为yyyy-MM-dd HH:mm:ss格式")
    
    # 验证数值列
    numeric_columns = ['HUFL', 'MUFL', 'LUFL']  # 验证这三个特征
    for col in numeric_columns:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            raise ValueError(f"{col}列包含非数值数据")

def process_dataset(df):
    """预处理数据集"""
    # 确保日期列格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 只保留需要的列
    df = df[['date', 'HUFL', 'MUFL', 'LUFL']]
    
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
        
        is_admin = (username == 'admin')
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            is_admin=is_admin
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
    try:
        # 检查请求体是否为空或不是JSON格式
        if not request.is_json:
            return jsonify({"error": "缺少JSON数据"}), 400
        
        username = request.json.get('username', None)
        password = request.json.get('password', None)
        
        # 验证必填字段
        if not username:
            return jsonify({"error": "请提供用户名"}), 400
        if not password:
            return jsonify({"error": "请提供密码"}), 400
            
        # 查询用户
        user = User.query.filter_by(username=username).first()
        
        # 验证用户和密码
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "用户名或密码不正确"}), 401
        
        # 更新最后登录时间为北京时间
        user.last_login = datetime.datetime.utcnow() + timedelta(hours=8)
        db.session.commit()
        
        # 创建访问令牌
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            "token": access_token,
            "username": user.username,
            "is_admin": user.is_admin
        }), 200
            
    except Exception as e:
        print(f"登录失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "登录处理时发生错误"}), 500

# 上传数据集
@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_dataset():
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"当前用户ID: {current_user_id}")
        
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
                    } for col in ['HUFL', 'MUFL', 'LUFL']
                }
            }
            
            # 保存处理后的数据
            temp_file = os.path.join(UPLOAD_DIR, f'temp_{current_user_id}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
            df.to_csv(temp_file, index=False)
            
            # 将DataFrame转换为JSON字符串存储
            dataset = Dataset(
                filename=file.filename,
                user_id=current_user_id,
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
            print(f"处理文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'文件处理错误: {str(e)}'}), 400
            
    except Exception as e:
        print(f"上传数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

def format_datetime(dt):
    """返回时间字符串，去掉微秒部分"""
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# 获取用户的数据集列表
@app.route('/datasets', methods=['GET'])
@jwt_required()
def get_datasets():
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取数据集列表，用户ID: {current_user_id}")
        
        datasets = Dataset.query.filter_by(user_id=current_user_id).all()
        return jsonify({
            'datasets': [{
                'id': ds.id,
                'filename': ds.filename,
                'upload_time': format_datetime(ds.upload_time)
            } for ds in datasets]
        }), 200
    except Exception as e:
        print(f"获取数据集列表时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取数据集列表失败: {str(e)}'}), 500

# 获取数据集详情
@app.route('/datasets/<int:dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id):
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取数据集详情，用户ID: {current_user_id}, 数据集ID: {dataset_id}")
        
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
        if not dataset:
            return jsonify({'error': '数据集不存在'}), 404
            
        return jsonify({
            'id': dataset.id,
            'filename': dataset.filename,
            'upload_time': format_datetime(dataset.upload_time),
            'data': dataset.data
        })
    except Exception as e:
        print(f"获取数据集详情时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取数据集详情失败: {str(e)}'}), 500

# 获取可用模型列表
@app.route('/api/models', methods=['GET'])
@jwt_required()
def get_models():
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
        verify_jwt_in_request()  # 添加此行来验证 JWT token
        current_user_id = get_jwt_identity()
        print(f"获取预测进度，用户ID: {current_user_id}, 预测ID: {prediction_id}")

        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此预测任务'}), 403
        
        progress = prediction_progress.get(prediction_id, {
            'status': 'unknown',
            'progress': 0,
            'message': '无法获取进度信息'
        })
        
        return jsonify(progress), 200
    except Exception as e:
        print(f"获取预测进度时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取预测进度失败: {str(e)}'}), 500

# 获取数据集的预测图片
@app.route('/prediction_images/<int:dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset_images(dataset_id):
    """获取指定数据集的预测图片"""
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取数据集预测图片，用户ID: {current_user_id}, 数据集ID: {dataset_id}")
        
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此数据集'}), 403

        # 构建图片目录路径
        dataset_results_dir = os.path.join(RESULTS_DIR, str(dataset_id))
        print(f"查找图片目录: {dataset_results_dir}")
        
        # 如果目录存在，递归列出所有文件
        images = []
        if os.path.exists(dataset_results_dir):
            print(f"目录存在，递归列出所有文件:")
            for root, dirs, files in os.walk(dataset_results_dir):
                for file in files:
                    if file.endswith('.png'):
                        rel_path = os.path.relpath(os.path.join(root, file), dataset_results_dir)
                        # 返回相对dataset_results_dir的路径，前端拼接时自动带子目录
                        images.append(rel_path.replace('\\', '/'))
            images.sort(reverse=True)
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
@jwt_required()
def delete_dataset(dataset_id):
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"删除数据集，用户ID: {current_user_id}, 数据集ID: {dataset_id}")
        
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权删除此数据集'}), 403
        
        try:
            # 删除预测结果目录
            results_dir = os.path.join(RESULTS_DIR, str(dataset_id))
            if os.path.exists(results_dir):
                print(f"删除预测结果目录: {results_dir}")
                try:
                    shutil.rmtree(results_dir)
                except Exception as e:
                    print(f"删除预测结果目录时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # 删除数据集文件
            dataset_file = os.path.join(UPLOAD_DIR, f'temp_{dataset_id}_*.csv')
            for file in glob.glob(dataset_file):
                try:
                    os.remove(file)
                    print(f"删除数据集文件: {file}")
                except Exception as e:
                    print(f"删除数据集文件时出错: {str(e)}")

            # 删除数据库记录（会自动删除相关的预测记录）
            db.session.delete(dataset)
            db.session.commit()
            
            return jsonify({'message': '数据集删除成功'}), 200
        except Exception as e:
            print(f"删除数据集过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return jsonify({'error': f'删除失败: {str(e)}'}), 500
    except Exception as e:
        print(f"删除数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

# 获取预测历史（返回所有历史预测记录）
@app.route('/predictions', methods=['GET'])
@jwt_required()
def get_predictions():
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取预测历史，用户ID: {current_user_id}")
        
        # 获取用户的所有数据集
        datasets = Dataset.query.filter_by(user_id=current_user_id).all()
        dataset_ids = [ds.id for ds in datasets]
        
        # 获取所有预测记录，按时间倒序排列
        all_predictions = Prediction.query.filter(Prediction.dataset_id.in_(dataset_ids))\
            .order_by(Prediction.prediction_time.desc()).all()
        
        # 构建响应数据
        predictions_data = []
        for pred in all_predictions:
            dataset = Dataset.query.get(pred.dataset_id)
            result = json.loads(pred.result) if pred.result else {}
            parameters = json.loads(pred.parameters) if pred.parameters else {}

            prediction_data = {
                'id': pred.id,
                'dataset_id': pred.dataset_id,
                'dataset_name': dataset.filename if dataset else 'Unknown',
                'model': parameters.get('model', 'Unknown'),
                'parameters': parameters,
                'status': result.get('status', 'unknown'),
                'created_at': format_datetime(pred.prediction_time),
                'result': result
            }
            predictions_data.append(prediction_data)
        
        return jsonify({
            'predictions': predictions_data
        }), 200
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
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
        verify_jwt_in_request()  # 添加此行来验证 JWT token
        current_user_id = get_jwt_identity()
        print(f"获取预测结果，用户ID: {current_user_id}, 预测ID: {prediction_id}")

        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此预测任务'}), 403
            
        result = json.loads(prediction.result) if prediction.result else {}
        parameters = json.loads(prediction.parameters) if prediction.parameters else {}
        
        return jsonify({
            'id': prediction.id,
            'dataset_id': prediction.dataset_id,
            'dataset_name': dataset.filename if dataset else 'Unknown',
            'model': parameters.get('model', 'Unknown'),
            'parameters': parameters,
            'status': result.get('status', 'unknown'),
            'created_at': format_datetime(prediction.prediction_time),
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
        verify_jwt_in_request()  # 添加此行来验证 JWT token
        current_user_id = get_jwt_identity()
        print(f"开始预测，用户ID: {current_user_id}")

        data = request.get_json()
        print("预测请求数据:", data)
        
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
            
        dataset_id = data.get('dataset_id')
        if not dataset_id:
            return jsonify({'error': '请选择数据集'}), 400
        
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此数据集'}), 403
        
        # 检查是否已有正在进行的预测任务
        existing_prediction = Prediction.query.filter_by(dataset_id=dataset_id).order_by(Prediction.prediction_time.desc()).first()
        if existing_prediction and json.loads(existing_prediction.result).get('status') == 'pending':
            return jsonify({'error': '该数据集已有正在进行的预测任务'}), 400

        # 将数据集保存为临时文件
        df = pd.read_json(dataset.data)
        temp_file_path = f'temp_{dataset_id}.csv'
        df.to_csv(temp_file_path, index=False)
        
        # 创建预测记录
        model_name = data.get('model_name', 'TimeXer6')  # 默认使用TimeXer6
        print("预测模型: ", model_name)
        pred_len = 6  # 默认预测长度
        
        # 根据选择的模型设置预测长度
        if model_name == 'TimeXer12':
            print("预测时长: 12小时")
            pred_len = 12
        elif model_name == 'TimeXer24':
            print("预测时长: 24小时")
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
        thread.daemon = True  # 设置为守护线程
        thread.start()
        
        return jsonify({
            'message': '预测任务已启动',
            'prediction_id': prediction.id
        }), 202
        
    except Exception as e:
        print(f"预测时出错: {str(e)}")
        import traceback
        traceback.print_exc()
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
@jwt_required()
def delete_prediction(prediction_id):
    """删除指定的预测结果"""
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"删除预测结果，用户ID: {current_user_id}, 预测ID: {prediction_id}")
        
        # 获取预测记录
        prediction = Prediction.query.get_or_404(prediction_id)
        dataset = Dataset.query.get(prediction.dataset_id)
        
        if not dataset:
            return jsonify({'error': '找不到相关数据集'}), 404
            
        if dataset.user_id != current_user_id:
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

# 获取特定特征的预测结果数据
@app.route('/prediction_data/<dataset_id>/<int:feature_index>', methods=['GET'])
@jwt_required()
def get_feature_prediction_data(dataset_id, feature_index):
    """获取指定数据集和特征的预测结果数据"""
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取预测结果数据，用户ID: {current_user_id}, 数据集ID: {dataset_id}, 特征索引: {feature_index}")
        
        # 验证用户是否有权限访问该数据集
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此数据集'}), 403

        # 获取预测结果数据
        results = PredictionResult.query.filter_by(
            dataset_name=str(dataset_id),
            feature_index=feature_index
        ).order_by(PredictionResult.time).all()
        
        if not results:
            return jsonify({'error': '未找到预测数据'}), 404
            
        # 将结果转换为列表格式
        data = [{
            'time': result.time,
            'prediction': result.prediction,
            'groundtruth': result.actual,
            'error': result.error
        } for result in results]
            
        feature_names = {
            0: 'HUFL (高压用电负荷)',
            1: 'MUFL (中压用电负荷)',
            2: 'LUFL (低压用电负荷)'
        }
            
        return jsonify({
            'status': 'success',
            'feature_name': feature_names.get(feature_index, f'特征 {feature_index}'),
            'data': data
        })
        
    except Exception as e:
        print(f"获取预测结果数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 获取所有特征的预测结果数据
@app.route('/prediction_data/<dataset_id>/all', methods=['GET'])
@jwt_required()
def get_all_features_prediction_data(dataset_id):
    """获取指定数据集的所有特征预测结果数据"""
    try:
        # 获取当前用户ID
        current_user_id = get_jwt_identity()
        print(f"获取所有特征预测结果数据，用户ID: {current_user_id}, 数据集ID: {dataset_id}")
        
        # 验证用户是否有权限访问该数据集
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此数据集'}), 403

        feature_names = {
            0: 'HUFL (高压用电负荷)',
            1: 'MUFL (中压用电负荷)',
            2: 'LUFL (低压用电负荷)'
        }
        
        # 获取所有特征的预测结果
        all_features_data = {}
        for feature_index in range(3):
            results = PredictionResult.query.filter_by(
                dataset_name=str(dataset_id),
                feature_index=feature_index
            ).order_by(PredictionResult.time).all()
            
            if results:
                data = [{
                    'time': result.time,
                    'prediction': result.prediction,
                    'groundtruth': result.actual,
                    'error': result.error
                } for result in results]
                
                all_features_data[feature_index] = {
                    'feature_name': feature_names.get(feature_index, f'特征 {feature_index}'),
                    'data': data
                }
            
        if not all_features_data:
            return jsonify({'error': '未找到预测数据'}), 404
            
        return jsonify({
            'status': 'success',
            'features': all_features_data
        })
        
    except Exception as e:
        print(f"获取所有特征预测结果数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 批量删除某个数据集下的所有预测结果
@app.route('/predictions/dataset/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_predictions_by_dataset(dataset_id):
    try:
        current_user_id = get_jwt_identity()
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权删除此数据集的预测结果'}), 403
        predictions = Prediction.query.filter_by(dataset_id=dataset_id).all()
        for pred in predictions:
            db.session.delete(pred)
        db.session.commit()
        return jsonify({'message': '该数据集下所有预测结果已删除'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

# 删除预测结果文件夹（即删除 results 目录下对应数据集的文件夹）
@app.route('/predictions/folder/<int:dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_prediction_folder(dataset_id):
    try:
        current_user_id = get_jwt_identity()
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权删除此数据集的预测结果文件夹'}), 403
        results_dir = os.path.join(RESULTS_DIR, str(dataset_id))
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
            return jsonify({'message': '预测结果文件夹已删除'}), 200
        else:
            return jsonify({'message': '预测结果文件夹不存在'}), 200
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

@app.errorhandler(NoAuthorizationError)
def handle_no_auth_error(e):
    print(f"JWT 授权失败: {str(e)}")
    return jsonify({'error': 'JWT 授权失败，请重新登录'}), 401

# ====== 清空 prediction_results 表 ======
@app.route('/clear_prediction_results', methods=['POST'])
def clear_prediction_results():
    try:
        num_deleted = PredictionResult.query.delete()
        db.session.commit()
        return {'message': f'已清空prediction_results表，删除记录数：{num_deleted}'}, 200
    except Exception as e:
        db.session.rollback()
        return {'error': str(e)}, 500

# 获取数据集前5行预览
@app.route('/datasets/<int:dataset_id>/preview', methods=['GET'])
@jwt_required()
def get_dataset_preview(dataset_id):
    try:
        current_user_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, user_id=current_user_id).first()
        if not dataset:
            return jsonify({'error': '数据集不存在'}), 404
        if not dataset.data:
            return jsonify({'error': '数据集内容为空'}), 400
        try:
            df = pd.read_json(dataset.data)
        except Exception as e:
            return jsonify({'error': f'数据解析失败: {str(e)}'}), 400
        preview = df.head(5).to_dict(orient='records')
        return jsonify({'preview': preview}), 200
    except Exception as e:
        print(f"获取数据集预览时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'获取数据集预览失败: {str(e)}'}), 500

# 新增接口：获取指定预测（子目录）下的图片
@app.route('/prediction_images_by_prediction/<int:dataset_id>/<string:sub_dir>', methods=['GET'])
@jwt_required()
def get_prediction_images_by_prediction(dataset_id, sub_dir):
    """只获取指定预测（子目录）下的图片"""
    try:
        current_user_id = get_jwt_identity()
        dataset = Dataset.query.get_or_404(dataset_id)
        if dataset.user_id != current_user_id:
            return jsonify({'error': '无权访问此数据集'}), 403
        # 构建子目录路径
        sub_dir_path = os.path.join(RESULTS_DIR, str(dataset_id), sub_dir)
        images = []
        if os.path.exists(sub_dir_path):
            for file in os.listdir(sub_dir_path):
                if file.endswith('.png'):
                    images.append(file)
            images.sort(reverse=True)
        return jsonify({'images': images, 'sub_dir': sub_dir}), 200
    except Exception as e:
        print(f"获取指定预测图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['GET'])
def get_comments():
    comments = Comment.query.order_by(Comment.created_at.asc()).all()
    def build_tree(parent_id=None):
        nodes = []
        for c in comments:
            if c.parent_id == parent_id:
                node = {
                    'id': c.id,
                    'user_id': c.user_id,
                    'username': c.username,
                    'content': c.content,
                    'created_at': c.created_at.isoformat(),
                    'parent_id': c.parent_id,
                    'children': build_tree(c.id)
                }
                nodes.append(node)
        return nodes
    return jsonify(build_tree(None))

@app.route('/comments', methods=['POST'])
@jwt_required()
def add_comment():
    data = request.json
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'error': '反馈内容不能为空'}), 400
    comment = Comment(user_id=user_id, username=user.username, content=content)
    db.session.add(comment)
    db.session.commit()
    return jsonify({'message': '评论成功'})

@app.route('/comments/<int:comment_id>', methods=['DELETE'])
@jwt_required()
def delete_comment(comment_id):
    user_id = get_jwt_identity()
    comment = Comment.query.get_or_404(comment_id)
    user = User.query.get(user_id)
    if comment.user_id != user_id and not user.is_admin:
        return jsonify({'error': '只能删除自己的评论'}), 403
    # 递归删除所有子评论
    def delete_children(parent):
        children = Comment.query.filter_by(parent_id=parent.id).all()
        for child in children:
            delete_children(child)
            db.session.delete(child)
    delete_children(comment)
    db.session.delete(comment)
    db.session.commit()
    return jsonify({'message': '评论已删除'})

@app.route('/comments/reply', methods=['POST'])
@jwt_required()
def reply_comment():
    data = request.json
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    content = data.get('content', '').strip()
    parent_id = data.get('parent_id')
    if not content:
        return jsonify({'error': '回复内容不能为空'}), 400
    parent_comment = Comment.query.get(parent_id)
    if not parent_comment:
        return jsonify({'error': '父评论不存在'}), 404
    comment = Comment(user_id=user_id, username=user.username, content=content, parent_id=parent_id)
    db.session.add(comment)
    db.session.commit()
    return jsonify({'message': '回复成功'})

@app.route('/user/info', methods=['GET'])
@jwt_required()
def get_user_info():
    """获取当前登录用户的信息"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        return jsonify({
            'username': user.username,
            'email': user.email,
            'role': 'admin' if user.is_admin else 'user',
            'createdAt': user.created_at.isoformat() if user.created_at else None,
            'lastLogin': user.last_login.isoformat() if user.last_login else None
        })
        
    except Exception as e:
        print(f"获取用户信息时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/user/stats', methods=['GET'])
@jwt_required()
def get_user_stats():
    """获取当前登录用户的统计数据"""
    try:
        current_user_id = get_jwt_identity()
        
        # 获取用户上传的数据集数量
        datasets_count = Dataset.query.filter_by(user_id=current_user_id).count()
        
        # 获取用户创建的预测任务数量
        # 首先获取用户的所有数据集ID
        user_datasets = Dataset.query.filter_by(user_id=current_user_id).all()
        user_dataset_ids = [ds.id for ds in user_datasets]
        
        # 然后查询这些数据集下的所有预测任务
        predictions_count = 0
        if user_dataset_ids:
            predictions_count = Prediction.query.filter(Prediction.dataset_id.in_(user_dataset_ids)).count()
        
        return jsonify({
            'datasetsCount': datasets_count,
            'predictionsCount': predictions_count
        })
        
    except Exception as e:
        print(f"获取用户统计数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/user/update', methods=['PUT'])
@jwt_required()
def update_user_info():
    """更新当前登录用户的信息"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        data = request.json
        
        # 更新邮箱
        if 'email' in data and data['email']:
            # 检查邮箱是否已被其他用户使用
            existing_user = User.query.filter(User.email == data['email'], User.id != current_user_id).first()
            if existing_user:
                return jsonify({'error': '该邮箱已被其他用户使用'}), 400
            user.email = data['email']
            
        # 更新密码
        if 'newPassword' in data and data.get('newPassword'):
            # 验证旧密码
            if not data.get('oldPassword'):
                return jsonify({'error': '请提供当前密码'}), 400
                
            if not check_password_hash(user.password_hash, data['oldPassword']):
                return jsonify({'error': '当前密码不正确'}), 400
                
            # 设置新密码
            user.password_hash = generate_password_hash(data['newPassword'])
            
        # 保存更改
        db.session.commit()
        
        return jsonify({'message': '用户信息更新成功'})
        
    except Exception as e:
        db.session.rollback()
        print(f"更新用户信息时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/users', methods=['GET'])
@jwt_required()
def admin_get_users():
    """仅管理员可用：获取所有用户信息"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if not user or not user.is_admin:
        return jsonify({'error': '无权限访问'}), 403

    users = User.query.all()
    user_list = []
    for u in users:
        user_list.append({
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'is_admin': u.is_admin,
            'created_at': u.created_at.isoformat() if u.created_at else None,
            'last_login': u.last_login.isoformat() if u.last_login else None
        })
    return jsonify({'users': user_list})

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def admin_delete_user(user_id):
    """仅管理员可用：删除指定用户"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if not user or not user.is_admin:
        return jsonify({'error': '无权限操作'}), 403
    if user_id == current_user_id:
        return jsonify({'error': '不能删除自己'}), 400
    target = User.query.get(user_id)
    if not target:
        return jsonify({'error': '用户不存在'}), 404
    db.session.delete(target)
    db.session.commit()
    return jsonify({'message': '用户已删除'})

@app.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@jwt_required()
def admin_reset_password(user_id):
    """仅管理员可用：重置指定用户密码"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if not user or not user.is_admin:
        return jsonify({'error': '无权限操作'}), 403
    data = request.get_json()
    new_password = data.get('password')
    if not new_password:
        return jsonify({'error': '新密码不能为空'}), 400
    target = User.query.get(user_id)
    if not target:
        return jsonify({'error': '用户不存在'}), 404
    target.password_hash = generate_password_hash(new_password)
    db.session.commit()
    return jsonify({'message': '密码已重置'})

@app.route('/admin/users/<int:user_id>/email', methods=['PUT'])
@jwt_required()
def admin_update_user_email(user_id):
    """仅管理员可用：修改指定用户邮箱，允许重复"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    if not user or not user.is_admin:
        return jsonify({'error': '无权限操作'}), 403
    data = request.get_json()
    new_email = data.get('email')
    if not new_email:
        return jsonify({'error': '邮箱不能为空'}), 400
    target = User.query.get(user_id)
    if not target:
        return jsonify({'error': '用户不存在'}), 404
    # 允许邮箱重复，去除唯一性校验
    target.email = new_email
    db.session.commit()
    return jsonify({'message': '邮箱已更新'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("Starting Flask application...")
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.methods} {rule}")
    app.run(debug=True, port=5000)
    fix_old_predictions() 