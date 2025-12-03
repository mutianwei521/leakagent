"""
意图识别分类器
基于Embedding向量相似度进行用户意图识别
"""
import os
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from .base_agent import BaseAgent

class IntentClassifier(BaseAgent):
    """基于Embedding的意图识别分类器"""
    
    def __init__(self):
        super().__init__("IntentClassifier")

        # 设置OpenAI API密钥和配置
        openai.api_base = "https://api.chatanywhere.tech"
        openai.api_key = "sk-eHk6ICs2KGZ2M2xJ0AZK9DJu3DVqgO91EnatH7FsUokii7HH"
        
        # 预定义的意图向量库
        self.intent_examples = {
            'hydraulic_simulation': [
                "进行水力计算",
                "运行水力模拟", 
                "计算管网压力",
                "分析流量分布",
                "模拟水力性能",
                "水力分析",
                "压力计算",
                "流量模拟",
                "hydraulic analysis",
                "pressure calculation",
                "flow simulation",
                "水力仿真",
                "管网仿真",
                "压力分析",
                "流速计算"
            ],
            'network_analysis': [
                "分析管网结构",
                "查看管网信息",
                "管网拓扑分析",
                "网络连通性",
                "管段统计",
                "节点分布",
                "管网概况",
                "网络结构",
                "拓扑结构",
                "管网组成"
            ],
            'partition_analysis': [
                "管网分区",
                "网络分区",
                "聚类分析",
                "FCM聚类",
                "模糊聚类",
                "分区分析",
                "区域划分",
                "管网划分",
                "节点聚类",
                "分区优化",
                "clustering",
                "partition",
                "分成几个区",
                "分成几个分区",
                "聚类成几个",
                "离群点检测",
                "outlier detection",
                "异常点检测",
                "分区可视化",
                "聚类可视化"
            ],
            'sensor_placement': [
                "传感器布置",
                "传感器优化",
                "压力监测点布置",
                "监测点优化",
                "传感器选择",
                "压力传感器",
                "监测点选择",
                "传感器配置",
                "监测网络优化",
                "传感器网络",
                "sensor placement",
                "sensor optimization",
                "pressure monitoring",
                "monitoring points",
                "sensor selection",
                "韧性分析",
                "传感器韧性",
                "故障分析",
                "检测覆盖率",
                "敏感度分析",
                "检测点布置",
                "检测点优化",
                "压力检测",
                "监测系统",
                "传感器系统"
            ],
            'leak_detection': [
                "漏损检测",
                "漏损分析",
                "泄漏检测",
                "泄漏分析",
                "漏水检测",
                "漏水分析",
                "异常检测",
                "故障检测",
                "漏损识别",
                "泄漏识别",
                "漏损定位",
                "泄漏定位",
                "漏损监测",
                "泄漏监测",
                "leak detection",
                "leak analysis",
                "leakage detection",
                "leakage analysis",
                "anomaly detection",
                "fault detection",
                "训练漏损模型",
                "训练检测模型",
                "漏损模型训练",
                "检测模型训练",
                "漏损机器学习",
                "漏损AI",
                "漏损预测",
                "泄漏预测",
                "异常预测",
                "故障预测",
                "漏损诊断",
                "泄漏诊断"
            ],
            'general_inquiry': [
                "这是什么文件",
                "文件内容介绍",
                "基本信息查询",
                "帮助信息",
                "使用说明",
                "功能介绍"
            ]
        }
        
        # 计算意图向量
        self.intent_embeddings = None
        self._compute_intent_embeddings()
    
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
            
            for intent, examples in self.intent_examples.items():
                embeddings = []
                
                for example in examples:
                    embedding = self._get_embedding(example)
                    if embedding is not None:
                        embeddings.append(embedding)
                
                if embeddings:
                    # 计算平均向量作为意图向量
                    self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
                    self.log_info(f"意图 '{intent}' 向量计算完成，样本数: {len(embeddings)}")
                else:
                    self.log_error(f"意图 '{intent}' 没有有效的embedding向量")
            
            self.log_info("所有意图向量计算完成")
            
        except Exception as e:
            self.log_error(f"计算意图向量失败: {e}")
            self.intent_embeddings = {}
    
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
