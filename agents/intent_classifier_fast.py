"""
快速意图识别分类器
使用缓存机制避免重复计算embedding向量
"""
import os
import json
import pickle
import hashlib
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from .base_agent import BaseAgent

class FastIntentClassifier(BaseAgent):
    """基于缓存的快速意图识别分类器"""
    
    def __init__(self):
        super().__init__("FastIntentClassifier")

        # 设置OpenAI API密钥和配置
        openai.api_base = "https://api.chatanywhere.tech"
        openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"
        
        # 缓存文件路径
        self.cache_dir = "cache"
        self.embeddings_cache_file = os.path.join(self.cache_dir, "intent_embeddings.pkl")
        self.examples_hash_file = os.path.join(self.cache_dir, "examples_hash.txt")
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 预定义的意图向量库（优化版 - 减少示例数量但保持准确性）
        self.intent_examples = {
            'hydraulic_simulation': [
                "进行水力计算", "运行水力模拟", "计算管网压力", "分析流量分布", "水力分析",
                "压力计算", "流量模拟", "水力仿真", "管网仿真", "压力分析"
            ],
            'network_analysis': [
                "分析管网结构", "查看管网信息", "管网拓扑分析", "管网概况", "网络结构",
                "拓扑结构", "管网组成", "节点分布", "管段统计"
            ],
            'partition_analysis': [
                "管网分区", "网络分区", "聚类分析", "分区分析", "区域划分",
                "分成几个区", "分成几个分区", "聚类成几个", "管网划分", "节点聚类",
                "分区优化", "离群点检测", "分区可视化", "FCM聚类", "模糊聚类"
            ],
            'sensor_placement': [
                "传感器布置", "传感器优化", "压力监测点布置", "监测点优化", "传感器选择",
                "压力传感器", "监测点选择", "传感器配置", "监测网络优化", "韧性分析",
                "检测覆盖率", "敏感度分析", "检测点布置", "压力检测"
            ],
            'leak_detection': [
                "漏损检测", "漏损分析", "泄漏检测", "异常检测", "训练漏损模型", "漏损模型训练",
                "故障检测", "漏损识别", "泄漏识别", "漏损定位", "漏损监测", "检测模型训练",
                "漏损机器学习", "漏损预测", "异常预测", "故障预测"
            ],
            'general_inquiry': [
                "这是什么文件", "文件内容介绍", "基本信息查询", "帮助信息", "使用说明", "功能介绍"
            ]
        }
        
        # 计算意图向量（使用缓存）
        self.intent_embeddings = None
        self._load_or_compute_embeddings()
    
    def _compute_examples_hash(self):
        """计算示例文本的哈希值，用于检测变化"""
        examples_str = json.dumps(self.intent_examples, sort_keys=True)
        return hashlib.md5(examples_str.encode()).hexdigest()
    
    def _should_recompute_embeddings(self):
        """检查是否需要重新计算embeddings"""
        if not os.path.exists(self.embeddings_cache_file):
            return True
        
        if not os.path.exists(self.examples_hash_file):
            return True
        
        # 检查示例是否有变化
        current_hash = self._compute_examples_hash()
        try:
            with open(self.examples_hash_file, 'r') as f:
                cached_hash = f.read().strip()
            return current_hash != cached_hash
        except:
            return True
    
    def _save_embeddings_cache(self):
        """保存embeddings到缓存文件"""
        try:
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.intent_embeddings, f)
            
            # 保存哈希值
            current_hash = self._compute_examples_hash()
            with open(self.examples_hash_file, 'w') as f:
                f.write(current_hash)
            
            self.log_info("Embeddings缓存已保存")
        except Exception as e:
            self.log_error(f"保存embeddings缓存失败: {e}")
    
    def _load_embeddings_cache(self):
        """从缓存文件加载embeddings"""
        try:
            with open(self.embeddings_cache_file, 'rb') as f:
                self.intent_embeddings = pickle.load(f)
            self.log_info("从缓存加载embeddings成功")
            return True
        except Exception as e:
            self.log_error(f"加载embeddings缓存失败: {e}")
            return False
    
    def _get_embedding(self, text: str):
        """获取文本的embedding向量"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            self.log_error(f"获取embedding失败: {e}")
            return None
    
    def _compute_intent_embeddings(self):
        """计算各个意图的embedding向量"""
        self.log_info("开始计算意图向量...")
        
        try:
            self.intent_embeddings = {}
            total_examples = sum(len(examples) for examples in self.intent_examples.values())
            processed = 0
            
            for intent, examples in self.intent_examples.items():
                embeddings = []
                
                for example in examples:
                    embedding = self._get_embedding(example)
                    if embedding is not None:
                        embeddings.append(embedding)
                    
                    processed += 1
                    if processed % 5 == 0:
                        self.log_info(f"进度: {processed}/{total_examples}")
                
                if embeddings:
                    # 计算平均向量作为意图向量
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    self.log_info(f"意图 '{intent}' 向量计算完成，样本数: {len(embeddings)}")
                else:
                    self.log_error(f"意图 '{intent}' 没有有效的embedding向量")
            
            self.log_info("所有意图向量计算完成")
            
            # 保存到缓存
            self._save_embeddings_cache()
            
        except Exception as e:
            self.log_error(f"计算意图向量失败: {e}")
            self.intent_embeddings = {}
    
    def _load_or_compute_embeddings(self):
        """加载或计算embeddings"""
        if self._should_recompute_embeddings():
            self.log_info("需要重新计算embeddings")
            self._compute_intent_embeddings()
        else:
            self.log_info("从缓存加载embeddings")
            if not self._load_embeddings_cache():
                self.log_info("缓存加载失败，重新计算")
                self._compute_intent_embeddings()
    
    def classify_intent(self, user_message: str):
        """分类用户意图"""
        if not self.intent_embeddings:
            self.log_error("意图向量未初始化")
            return {
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'all_similarities': {},
                'error': '意图向量未初始化'
            }
        
        try:
            # 获取用户消息的embedding
            user_embedding = self._get_embedding(user_message)
            if user_embedding is None:
                return {
                    'intent': 'general_inquiry',
                    'confidence': 0.0,
                    'all_similarities': {},
                    'error': '无法获取用户消息的embedding'
                }
            
            # 计算与各个意图的相似度
            similarities = {}
            user_embedding = user_embedding.reshape(1, -1)
            
            for intent, intent_vector in self.intent_embeddings.items():
                intent_vector = intent_vector.reshape(1, -1)
                similarity = cosine_similarity(user_embedding, intent_vector)[0][0]
                similarities[intent] = float(similarity)
            
            # 返回最高相似度的意图和置信度
            best_intent = max(similarities, key=similarities.get)
            confidence = similarities[best_intent]
            
            self.log_info(f"意图识别结果: {best_intent} (置信度: {confidence:.3f})")
            
            return {
                'intent': best_intent,
                'confidence': confidence,
                'all_similarities': similarities
            }
            
        except Exception as e:
            self.log_error(f"意图识别失败: {e}")
            return {
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'all_similarities': {},
                'error': str(e)
            }
    
    def process(self, user_message: str):
        """处理用户消息，返回意图识别结果"""
        return self.classify_intent(user_message)
