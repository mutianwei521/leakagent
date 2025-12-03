"""
基础智能体类
"""
import os
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(f"agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """处理输入的抽象方法"""
        pass
    
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.name}] {message}")

    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.name}] {message}")

    def log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.name}] {message}")
    
    def save_result(self, result: dict, filename: str):
        """保存处理结果"""
        try:
            os.makedirs('logs', exist_ok=True)
            filepath = os.path.join('logs', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.log_info(f"结果已保存到: {filepath}")
            return filepath
        except Exception as e:
            self.log_error(f"保存结果失败: {e}")
            return None
    
    def get_status(self):
        """获取智能体状态"""
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'status': 'active'
        }
