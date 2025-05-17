from app import app, db, User
from werkzeug.security import generate_password_hash

def init_db():
    """初始化数据库并创建测试用户"""
    with app.app_context():
        # 创建所有表
        print("创建数据库表...")
        db.create_all()
        
        # 创建测试用户
        test_users = [
            {'username': 'admin', 'password': 'admin123'},
            {'username': 'test', 'password': 'test123'},
            {'username': '13937208182', 'password': '123456'}
        ]
        
        for user_data in test_users:
            # 检查用户是否已存在
            if not User.query.filter_by(username=user_data['username']).first():
                user = User(
                    username=user_data['username'],
                    password_hash=generate_password_hash(user_data['password'])
                )
                db.session.add(user)
                print(f"创建用户: {user_data['username']}")
        
        # 提交更改
        try:
            db.session.commit()
            print("数据库初始化完成！")
            print("\n可用的测试账号：")
            for user in test_users:
                print(f"用户名: {user['username']}, 密码: {user['password']}")
        except Exception as e:
            print(f"初始化数据库时出错: {str(e)}")
            db.session.rollback()

if __name__ == '__main__':
    init_db() 