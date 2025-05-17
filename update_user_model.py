import sqlite3
import os
from datetime import datetime

# 获取数据库路径
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_forecast.db')

def migrate_user_table():
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查email字段是否已存在
        cursor.execute("PRAGMA table_info(user)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        # 添加新字段
        if 'email' not in column_names:
            print("添加 email 字段...")
            cursor.execute("ALTER TABLE user ADD COLUMN email TEXT")
        
        if 'created_at' not in column_names:
            print("添加 created_at 字段...")
            current_time = datetime.utcnow().isoformat()
            cursor.execute(f"ALTER TABLE user ADD COLUMN created_at TIMESTAMP DEFAULT '{current_time}'")
        
        if 'last_login' not in column_names:
            print("添加 last_login 字段...")
            cursor.execute("ALTER TABLE user ADD COLUMN last_login TIMESTAMP")
            
        # 提交更改
        conn.commit()
        print("数据库迁移成功！")
        
    except Exception as e:
        print(f"迁移出错: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    print(f"正在更新数据库: {db_path}")
    migrate_user_table() 