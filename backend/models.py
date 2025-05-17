from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

db = SQLAlchemy()

def beijing_now():
    return datetime.utcnow() + timedelta(hours=8)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    datasets = db.relationship('Dataset', backref='user', lazy=True, cascade='all, delete-orphan')

class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    upload_time = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    data = db.Column(db.Text, nullable=False)
    predictions = db.relationship('Prediction', backref='dataset', lazy=True, cascade='all, delete-orphan')

class Prediction(db.Model):
    __tablename__ = 'prediction'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id', ondelete='CASCADE'), nullable=False)
    prediction_time = db.Column(db.DateTime(timezone=True), nullable=False, default=datetime.now)
    result = db.Column(db.Text, nullable=False)
    parameters = db.Column(db.Text, nullable=False)

class PredictionResult(db.Model):
    __tablename__ = 'prediction_results'
    id = db.Column(db.Integer, primary_key=True)
    dataset_name = db.Column(db.String(255), nullable=False)
    feature_index = db.Column(db.Integer, nullable=False)
    time = db.Column(db.String(50), nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    actual = db.Column(db.Float, nullable=False)
    error = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'dataset_name': self.dataset_name,
            'feature_index': self.feature_index,
            'time': self.time,
            'prediction': self.prediction,
            'actual': self.actual,
            'error': self.error,
            'created_at': self.created_at.isoformat()
        }

class Comment(db.Model):
    __tablename__ = 'comment'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    username = db.Column(db.String(80), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=True)
    # 冗余字段，便于前端显示 