#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
供水管网漏损检测系统主程序
Water Distribution Network Leak Detection System using LTFM
"""

import os
import sys
import yaml
import argparse
import pandas as pd
from loguru import logger
from typing import Dict, List

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.epanet_handler import EPANETHandler
from src.data.sensitivity_analyzer import SensitivityAnalyzer
from src.utils.fcm_partitioner import FCMPartitioner
from src.models.graph2vec_encoder import Graph2VecEncoder
from src.training.trainer import LTFMTrainer
from src.inference.predictor import LTFMPredictor


def load_config(config_path: str = "config.yaml") -> Dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def setup_logging(config: Dict):
    """设置日志"""
    try:
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_format = config.get('logging', {}).get('format', 
                               "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        # 创建日志目录
        os.makedirs(config.get('data', {}).get('log_dir', 'logs'), exist_ok=True)
        
        # 配置loguru
        logger.remove()  # 移除默认处理器
        logger.add(sys.stderr, level=log_level, format=log_format)
        logger.add(
            os.path.join(config.get('data', {}).get('log_dir', 'logs'), "system.log"),
            level=log_level,
            format=log_format,
            rotation="1 day",
            retention="30 days"
        )
        
        logger.info("日志系统初始化完成")
        
    except Exception as e:
        print(f"日志设置失败: {e}")


def train_mode(config: Dict, args):
    """训练模式"""
    try:
        logger.info("开始训练模式")
        
        # 1. 初始化EPANET处理器
        epanet_file = args.epanet_file or config['data']['epanet_file']
        epanet_handler = EPANETHandler(epanet_file)
        
        if not epanet_handler.load_network():
            logger.error("EPANET网络加载失败")
            return False
        
        # 2. 计算压力灵敏度
        logger.info("计算压力灵敏度...")
        sensitivity_analyzer = SensitivityAnalyzer(epanet_handler)
        
        if not sensitivity_analyzer.calculate_normal_pressure(
            config['hydraulic']['simulation_hours'],
            config['hydraulic']['time_step']
        ):
            logger.error("正常压力计算失败")
            return False
        
        if not sensitivity_analyzer.calculate_sensitivity_matrix(
            config['hydraulic']['demand_reduction_ratio'],
            config['hydraulic']['simulation_hours'],
            config['hydraulic']['time_step']
        ):
            logger.error("灵敏度矩阵计算失败")
            return False
        
        sensitivity_analyzer.normalize_sensitivity_matrix()
        node_weights = sensitivity_analyzer.calculate_node_weights()
        
        # 保存灵敏度数据
        sensitivity_analyzer.save_sensitivity_data(config['data']['output_dir'])
        
        # 3. FCM分区
        logger.info("进行FCM分区...")
        fcm_partitioner = FCMPartitioner(
            n_clusters=config['fcm']['n_clusters'],
            m=config['fcm']['m'],
            max_iter=config['fcm']['max_iter'],
            error=config['fcm']['error']
        )
        
        # 准备特征
        network_graph = epanet_handler.get_network_graph()
        pipe_lengths = epanet_handler.get_pipe_lengths()
        
        features = fcm_partitioner.prepare_features(
            epanet_handler.adjacency_matrix,
            node_weights,
            pipe_lengths,
            epanet_handler.pipe_names,
            epanet_handler.node_names
        )
        
        if not fcm_partitioner.partition_network(features):
            logger.error("FCM分区失败")
            return False

        # 离群点剔除
        logger.info("进行离群点检测和处理...")
        # 获取原始分区标签（从0开始，需要转换为从1开始）
        raw_labels = fcm_partitioner.partition_labels + 1

        # 获取所有节点名称（包括非需水节点）
        all_node_names = list(epanet_handler.wn.node_name_list)

        # 执行离群点检测和处理（保存可视化）
        partition_viz_dir = os.path.join(config['data']['output_dir'], 'partition_visualization')
        refined_labels = fcm_partitioner.remove_outliers_iteratively(
            wn=epanet_handler.wn,
            nodes=all_node_names,
            demands=epanet_handler.node_names,
            raw_labels=raw_labels,
            k_nearest=config['fcm'].get('k_nearest', 10),
            outliers_detection=config['fcm'].get('outliers_detection', True),
            seed=config['fcm'].get('seed', 42),
            output_dir=partition_viz_dir
        )

        # 更新分区标签（转换回从0开始）
        fcm_partitioner.partition_labels = refined_labels - 1

        # 保存分区结果
        fcm_partitioner.save_partition_results(
            config['data']['output_dir'],
            epanet_handler.node_names
        )
        
        # 4. 训练Graph2Vec
        logger.info("训练Graph2Vec...")
        subgraphs = fcm_partitioner.get_partition_subgraphs(network_graph, epanet_handler.node_names)
        all_graphs = [network_graph] + subgraphs
        
        graph2vec_encoder = Graph2VecEncoder(
            embedding_dim=int(config['graph2vec']['dimensions']),
            workers=int(config['graph2vec']['workers']),
            epochs=int(config['graph2vec']['epochs']),
            min_count=int(config['graph2vec']['min_count']),
            learning_rate=float(config['graph2vec']['learning_rate'])
        )
        
        if not graph2vec_encoder.train_model(all_graphs):
            logger.error("Graph2Vec训练失败")
            return False
        
        # 保存Graph2Vec模型
        graph2vec_encoder.save_model(os.path.join(config['data']['output_dir'], 'graph2vec_model.model'))
        
        # 5. 训练LTFM模型
        logger.info("训练LTFM模型...")
        trainer = LTFMTrainer(config)
        
        if not trainer.initialize_models():
            logger.error("LTFM模型初始化失败")
            return False
        
        # 设置预训练的Graph2Vec
        trainer.graph2vec_encoder = graph2vec_encoder
        
        # 生成训练场景
        train_scenarios = trainer.generate_training_scenarios(
            epanet_handler, sensitivity_analyzer, fcm_partitioner,
            n_scenarios=args.n_scenarios or 1000
        )
        
        # 使用分层采样分割训练和验证集，确保验证集包含正常和异常样本
        normal_scenarios = [s for s in train_scenarios if s['global_label'] == 0]
        anomaly_scenarios = [s for s in train_scenarios if s['global_label'] == 1]

        logger.info(f"正常场景数量: {len(normal_scenarios)}, 异常场景数量: {len(anomaly_scenarios)}")

        # 分别对正常和异常场景进行8:2分割
        normal_split = int(len(normal_scenarios) * 0.8)
        anomaly_split = int(len(anomaly_scenarios) * 0.8)

        train_data = normal_scenarios[:normal_split] + anomaly_scenarios[:anomaly_split]
        val_data = normal_scenarios[normal_split:] + anomaly_scenarios[anomaly_split:]

        # 打乱数据
        import random
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        # 训练模型
        if not trainer.train(train_data, val_data, skip_stage1=args.skip_stage1):
            logger.error("LTFM模型训练失败")
            return False
        
        # 6. Stage 2: 训练NodeLocalizer（节点级漏损定位）
        logger.info("开始Stage 2: 训练NodeLocalizer...")
        if not trainer.train_node_localizer(train_data, val_data):
            logger.warning("NodeLocalizer训练失败，但LTFM模型已训练完成")
        
        logger.info("训练完成")
        return True
        
    except Exception as e:
        logger.error(f"训练模式失败: {e}")
        return False


def inference_mode(config: Dict, args):
    """推理模式"""
    try:
        logger.info("开始推理模式")
        
        # 初始化预测器
        predictor = LTFMPredictor(config)
        
        # 加载模型
        graph2vec_path = args.graph2vec_model or os.path.join(config['data']['output_dir'], 'graph2vec_model.model')
        ltfm_checkpoint_path = args.ltfm_checkpoint or os.path.join('output/checkpoints', 'best_model.pth')
        
        if not predictor.load_models(graph2vec_path, ltfm_checkpoint_path):
            logger.error("模型加载失败")
            return False
        
        # 初始化网络
        epanet_file = args.epanet_file or config['data']['epanet_file']
        if not predictor.initialize_network(epanet_file):
            logger.error("网络初始化失败")
            return False
        
        # 显示网络状态
        status = predictor.get_network_status()
        logger.info(f"网络状态: {status}")
        
        if args.test_data:
            # 批量预测模式
            logger.info(f"加载测试数据: {args.test_data}")
            test_pressure_data = pd.read_csv(args.test_data, index_col=0, encoding='utf-8')
            
            # 这里假设测试数据是单个时间点的压力数据
            # 实际使用时可能需要根据数据格式调整
            results = predictor.batch_predict([test_pressure_data])
            
            # 保存结果
            output_path = args.output or os.path.join(config['data']['output_dir'], 'prediction_results.csv')
            predictor.save_prediction_results(results, output_path)
            
            # 显示结果
            for i, result in enumerate(results):
                logger.info(f"样本 {i}: 异常={result.get('global_anomaly', False)}, "
                          f"概率={result.get('global_probability', 0):.3f}, "
                          f"区域={result.get('anomaly_region', 0)}")
        else:
            # 实时监控模式
            logger.info("实时监控模式（演示）")
            logger.info("在实际应用中，这里会连接到实时数据源")

            # 演示：使用正常压力数据进行预测
            result = predictor.predict_anomaly(predictor.normal_pressure)
            logger.info(f"演示预测结果: {result}")

            # 显示分区信息
            if result and 'partitions' in result:
                logger.info(f"FCM分区结果: {len(result['partitions'])} 个分区")
                for i, partition in enumerate(result['partitions']):
                    logger.info(f"分区 {i}: {len(partition)} 个节点")
        
        return True
        
    except Exception as e:
        logger.error(f"推理模式失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="供水管网漏损检测系统")
    parser.add_argument('--mode', choices=['train', 'inference'], 
                       help='运行模式：train（训练）或 inference（推理）', default='train')
    parser.add_argument('--config', default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--epanet-file', 
                       help='EPANET网络文件路径')
    
    # 训练模式参数
    parser.add_argument('--n-scenarios', type=int,
                       help='训练场景数量')
    parser.add_argument('--skip-stage1', action='store_true',
                       help='跳过Stage 1 (LTFM) 训练，直接加载已有模型训练Stage 2')
    
    # 推理模式参数
    parser.add_argument('--graph2vec-model',
                       help='Graph2Vec模型路径')
    parser.add_argument('--ltfm-checkpoint',
                       help='LTFM检查点路径')
    parser.add_argument('--test-data',
                       help='测试数据文件路径')
    parser.add_argument('--output',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        print("配置加载失败，退出程序")
        return 1
    
    # 设置日志
    setup_logging(config)
    
    # 创建输出目录
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    
    # 根据模式运行
    if args.mode == 'train':
        success = train_mode(config, args)
    elif args.mode == 'inference':
        success = inference_mode(config, args)
    else:
        logger.error(f"未知模式: {args.mode}")
        return 1
    
    if success:
        logger.info("程序执行成功")
        return 0
    else:
        logger.error("程序执行失败")
        return 1


if __name__ == "__main__":
    # sys.exit(main())
    main()
