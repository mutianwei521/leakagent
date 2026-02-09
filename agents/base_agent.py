"""
Base Agent Class
"""
import os
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Agent base class"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger"""
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
        """Abstract method to process input"""
        pass
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.name}] {message}")

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.name}] {message}")
    
    def save_result(self, result: dict, filename: str):
        """Save processing result"""
        try:
            os.makedirs('logs', exist_ok=True)
            filepath = os.path.join('logs', filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.log_info(f"Result saved to: {filepath}")
            return filepath
        except Exception as e:
            self.log_error(f"Failed to save result: {e}")
            return None
    
    def get_status(self):
        """Get agent status"""
        return {
            'name': self.name,
            'created_at': self.created_at.isoformat(),
            'status': 'active'
        }
