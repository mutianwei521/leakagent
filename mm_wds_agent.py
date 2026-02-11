import os
import wntr
import networkx as nx
import json
import sys

# Ensure local modules are findable
sys.path.append(os.getcwd())

from typing import List, Optional, Dict, Any
from optimization_utils.objectives import calculate_fef, calculate_nr, run_simulation

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# API Configuration
# Please set your API keys in the .env file
# You can copy .env.example to .env and fill in your keys
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  WARNING: OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    print("   Example: OPENAI_API_KEY=sk-...")

# Optional: Set custom API base URL if needed
# os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

from optimization_utils.rag_manager import ingest_inp_file, retrieve_knowledge

@tool
def hydraulic_inspector(inp_file: str, query: str):
    """
    水力检查工具。
    当用户询问水力属性、节点数量、管道数量或模拟结果时，请使用此工具。
    不要盲目运行模拟。此工具会首先检查知识图谱。
    """
    try:
        # 摄取/检索数据
        data = retrieve_knowledge(inp_file, query_type="summary")
        if "error" in data:
            return json.dumps({"status": "error", "error": data["error"]})

        # 将结构化摘要直接返回给大模型（LLM）
        # 大模型将解析 JSON 以回答用户的问题（例如“计算节点数量”）。
        return json.dumps({
            "status": "success",
            "source": "GraphRAG Cache",
            "file_info": data["filename"],
            "statistics": data["stats"],
            "hydraulics": data["hydraulics"]
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@tool
def reliability_assessor(inp_file: str):
    """
    可靠性评估工具。
    计算流量熵（FEF）和管网弹性（NR）。
    """
    try:
        wn = wntr.network.WaterNetworkModel(inp_file)
        results = run_simulation(wn) 
        fef = calculate_fef(wn, results)
        try:
             nr = calculate_nr(wn, results)
        except:
             nr = 0.5 
             
        return json.dumps({
            "metric_type": "Reliability",
            "FEF": fef,
            "NR": nr,
            "interpretation": "Values > 0.6 indicate high resilience." if fef > 0.6 else "Values < 0.5 indicate redundancy deficit."
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@tool
def graph_rag_retriever(inp_file: str, entity_id: str):
    """
    拓扑语义 GraphRAG 工具。
    从知识图谱中检索关于节点或链路（例如 'J-10' 或 'Pipe-1'）的具体细节。
    当用户询问特定元素时使用此工具。
    """
    try:
        # 传递 query_type="entity" 以获取详细数据
        data = retrieve_knowledge(inp_file, query_type="entity", entity_id=entity_id)
        
        if "error" in data:
            return json.dumps({"status": "error", "error": data["error"]})
            
        return json.dumps({
            "status": "found",
            "entity": entity_id,
            "details": data
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@tool
def network_partitioner(inp_file: str, num_partitions: Optional[int] = None, algorithm: Optional[str] = None):
    """
    管网分区工具。
    当用户要求将管网“分区”或“划分”为区域/社区时，请使用此工具。
    
    算法选择：
    - 默认：“louvain”（使用模块化优化的社区检测）
    - 备选：“fcm”（基于压力敏感性的模糊 C 均值聚类）
    
    当用户明确提到“FCM”、“模糊”、“模糊 C 均值”、“基于敏感性”或“压力敏感性”分区时使用 FCM。
    
    重要 - “num_partitions” 如何工作：
    - Louvain：产生离散的分区计数；如果目标较低，则合并社区。
    - FCM：直接使用指定的聚类数量。
    
    如果用户指定了数量（例如，“5 个区域”，“划分为 3 个”），请将其作为 'num_partitions' 传递。
    仅在用户明确请求基于 FCM 的分区时传递 algorithm="fcm"。
    """
    try:
        # 默认为 Louvain 算法
        use_fcm = algorithm and algorithm.lower() in ['fcm', 'fuzzy', 'fuzzy-c-means', 'fuzzycmeans']
        
        if use_fcm:
            # 使用 FCM 分区
            from partition_utils.fcm_partition import run_fcm_partitioning_for_agent
            
            result = run_fcm_partitioning_for_agent(
                inp_file, 
                num_partitions=num_partitions or 5,
                fuzziness=1.5
            )
            
            if result["status"] == "error":
                return json.dumps({"status": "error", "error": result["error"]})
            
            # 格式化针对 FCM 的响应
            base_url = "http://127.0.0.1:5000"
            
            response_text = f"## ✅ {result.get('msg', 'FCM partitioning completed.')}\n\n"
            response_text += "---\n\n"
            response_text += "### 📊 FCM Partitioning Results\n\n"
            
            # 分区统计表
            response_text += "| Partition | Node Count |\n"
            response_text += "|-----------|------------|\n"
            for partition, count in result['partition_stats'].items():
                response_text += f"| {partition} | {count} |\n"
            response_text += "\n"
            
            # 指标
            response_text += "### 📈 Clustering Metrics\n\n"
            response_text += f"- **Fuzzy Partition Coefficient (FPC):** {result['metrics']['fpc']:.4f}\n"
            response_text += f"- **Convergence Iterations:** {result['metrics']['iterations']}\n"
            response_text += f"- **Fuzziness Parameter (m):** {result['fuzziness']}\n\n"
            
            # 可视化
            if result.get('viz_file'):
                viz_filename = os.path.basename(result['viz_file'])
                response_text += "### 🖼️ Visualization\n\n"
                response_text += f"![FCM Partition]({base_url}/partition_results/{viz_filename})\n\n"
                response_text += f"[Download Visualization]({base_url}/partition_results/{viz_filename})\n\n"
            
            # Summary JSON
            if result.get('summary_json'):
                response_text += f"📄 [Download Summary JSON]({base_url}/{result['summary_json']})\n\n"
            
            response_text += "---\n\n"
            response_text += "### 🔧 Recommended Next Steps\n\n"
            response_text += "1. **Review partition boundaries** - Check that zones are spatially coherent\n"
            response_text += "2. **Analyze boundary pipes** - Use boundary_analyzer for valve placement\n"
            response_text += "3. **Place sensors** - Use sensor_placer for optimal monitoring points\n"
            
            return json.dumps({
                "status": "success", 
                "msg": response_text,
                "algorithm": "FCM",
                "raw_data": result
            }, ensure_ascii=False)
            
        else:
            # 使用 Louvain 算法（默认）
            from optimization_utils.partition_manager import run_partitioning_for_agent
            
            result = run_partitioning_for_agent(inp_file, target_k=num_partitions)
            
            if result["status"] == "error":
                 return json.dumps({"status": "error", "error": result["error"]})
            
            # 使用增强的 Markdown 格式化用户友好的响应
            base_url = "http://127.0.0.1:5000"
            
            response_text = f"## ✅ {result.get('msg', 'Partitioning completed.')}\n\n"
            response_text += "---\n\n"
            response_text += "### 📊 Output Files\n\n"
            
            # 图像
            for plot_path in result['plots']:
                 filename = os.path.basename(plot_path)
                 label = filename.replace(".png", "").replace("_", " ").title()
                 response_text += f"**Visualization:**\n\n"
                 response_text += f"![{label}]({base_url}/{plot_path})\n\n"
                 response_text += f"🖼️ [Download Image]({base_url}/{plot_path})\n\n"
                 
            # Summary JSON
            response_text += f"📄 [Download Summary JSON]({base_url}/{result['summary_json']})\n\n"
            
            response_text += "---\n\n"
            response_text += "### 🔧 Recommended Next Steps\n\n"
            response_text += "1. **Review partition boundaries** - Check that zones are geographically sensible\n"
            response_text += "2. **Verify hydraulic performance** - Run simulation for each zone\n"
            response_text += "3. **Identify isolation valves** - Locate boundary pipes for valve placement\n"
            
            return json.dumps({
                "status": "success", 
                "msg": response_text,
                "algorithm": "Louvain",
                "raw_data": result
            }, ensure_ascii=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)})


@tool
def boundary_analyzer(inp_file: str, num_partitions: Optional[int] = None):
    """
    分析分区之间的边界管道。
    当用户询问“边界管道”、“切割边”、“隔离阀”、“区域之间的连接”或“边界管段”时，请使用此工具。
    
    该工具会自动根据 INP 文件查找分区结果。
    无需单独上传 partition_summary.json。
    
    如果未指定 num_partitions，它将使用现有的分区结果。
    """
    try:
        from optimization_utils.zone_optimizer import analyze_boundary_pipes
        
        result = analyze_boundary_pipes(inp_file, target_k=num_partitions)
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
        
        # 使用增强的 Markdown 格式化响应
        base_url = "http://127.0.0.1:5000"
        
        response_text = f"## 🔗 Boundary Pipe Analysis ({result['partition_count']} Zones)\n\n"
        response_text += f"**Total Boundary Pipes:** {result['boundary_pipe_count']}\n\n"
        response_text += "---\n\n"
        response_text += "### 📋 Boundary Pipe List\n\n"
        response_text += "| Pipe | From Node | To Node | Zone→Zone | Diameter (mm) | Length (m) |\n"
        response_text += "|------|-----------|---------|-----------|---------------|------------|\n"
        
        for p in result['boundary_pipes'][:20]:  # 为了可读性限制为 20 个
            response_text += f"| {p['pipe']} | {p['from_node']} | {p['to_node']} | {p['zone_from']}→{p['zone_to']} | {p['diameter_mm']} | {p['length_m']} |\n"
        
        if result['boundary_pipe_count'] > 20:
            response_text += f"\n*...and {result['boundary_pipe_count'] - 20} more pipes*\n"
        
        response_text += "\n---\n\n"
        response_text += "### 🔧 Engineering Recommendations\n\n"
        response_text += "1. **Isolation Valves**: Install valves on boundary pipes to enable zone isolation\n"
        response_text += "2. **Flow Meters**: Consider metering at zone entry points for leak detection\n"
        response_text += "3. **Prioritize**: Focus on larger diameter pipes for valve placement\n"
        
        return json.dumps({
            "status": "success",
            "msg": response_text,
            "raw_data": result
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@tool
def zone_optimizer(inp_file: str, num_partitions: Optional[int] = None, 
                   pop_size: int = 20, n_gen: int = 50):
    """
    对分区边界配置运行 NSGA-II 优化。
    目标：最大化 FEF、HRE、MRE、NR；最小化开启管道。
    当用户要求“优化”、“改进”或寻找“最佳配置”以获得特定数量的区域时，请使用此工具。
    
    该工具针对单一分区计数（例如 10 个区域）优化边界管道的阀门状态（开启/关闭）。
    它不会寻找最佳分区数量。
    """
    try:
        from optimization_utils.zone_optimizer import run_zone_optimization
        
        # 除非用户另有指定，否则默认使用低迭代次数以保证交互速度
        result = run_zone_optimization(
            inp_file, 
            target_k=num_partitions,
            pop_size=pop_size, 
            n_gen=n_gen
        )
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
            
        # 增强的 Markdown 输出
        base_url = "http://127.0.0.1:5000"
        obj = result['best_objectives']
        
        response_text = f"## 🚀 Optimization Complete ({result.get('msg', '').split('for ')[-1]})\n\n"
        response_text += "---\n\n"
        response_text += "### 🏆 Best Solution Objectives\n\n"
        
        response_text += f"| Objective | Value | Description |\n"
        response_text += f"|-----------|-------|-------------|\n"
        response_text += f"| **FEF** | {obj['FEF']:.4f} | Flow Entropy (Reliability) |\n"
        response_text += f"| **HRE** | {obj['HRE']:.4f} | Hydraulic Resilience |\n"
        response_text += f"| **MRE** | {obj['MRE']:.4f} | Mechanical Reliability |\n"
        response_text += f"| **NR** | {obj['NR']:.4f} | Network Resilience |\n"
        response_text += f"| **Open Valves** | {obj['open_pipes']} | Boundary Connections |\n\n"
        
        response_text += f"📄 [Download Optimization Results JSON]({base_url}/{result['optimization_file']})\n\n"
        response_text += "---\n"
        response_text += "### 💡 Interpretation\n"
        response_text += f"- **Values Optimized**: {result['boundary_count']} boundary valve settings found.\n"
        response_text += "- **Balance**: Solution balances reliability (Entropy) with isolation (Closed Valves).\n"
        
        return json.dumps({
            "status": "success",
            "msg": response_text,
            "raw_data": result
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

@tool
def visual_analyzer(inp_file: str, analysis_type: Optional[str] = "combined"):
    """
    视觉感知分析工具。
    当用户要求“可视化”、“显示热力图”、“分析压力分布”、“显示流量模式”、“视觉分析”或“生成管网图”时，请使用此工具。
    
    该工具生成视觉热图并从管网中提取视觉特征：
    - 压力热图（蓝色=低压，红色=高压）
    - 流量可视化（线宽 = 速度）
    - 拓扑异常检测（末端、桥接）
    
    analysis_type 选项：
    - "pressure": 仅压力热图
    - "flow": 仅流量可视化  
    - "combined": 压力和流量（默认）
    - "features": 提取视觉特征而不生成图像
    """
    try:
        from partition_utils.visual_perception import (
            analyze_network_visually,
            extract_visual_features,
            get_vlm_prompt_template
        )
        import wntr
        
        base_url = "http://127.0.0.1:5000"
        
        # 运行视觉分析
        result = analyze_network_visually(inp_file)
        
        if "error" in result:
            return json.dumps({"status": "error", "error": result["error"]})
        
        # Format response with enhanced markdown
        response_text = f"## 🎨 Visual Analysis Complete: {result['network_name']}\n\n"
        response_text += f"**Network Size:** {result['node_count']} nodes, {result['link_count']} links\n\n"
        response_text += "---\n\n"
        
        # 热图图像
        response_text += "### 📊 Generated Visualizations\n\n"
        
        for viz_type, path in result['heatmap_paths'].items():
            if analysis_type == "combined" or analysis_type == viz_type:
                filename = os.path.basename(path)
                label = viz_type.replace("_", " ").title()
                # 使用相对路径进行网页显示
                rel_path = path.replace("\\", "/")
                response_text += f"**{label} Heatmap:**\n\n"
                response_text += f"![{label}]({base_url}/visual_outputs/{filename})\n\n"
                response_text += f"🖼️ [Download {label} Image]({base_url}/visual_outputs/{filename})\n\n"
        
        # 视觉特征
        features = result['visual_features']
        response_text += "---\n\n"
        response_text += "### 🔍 Extracted Visual Features\n\n"
        
        # 拓扑异常
        topo = features.get('topological_anomalies', {})
        response_text += "**Topological Analysis:**\n"
        response_text += f"- 🌉 Bridge Nodes (Critical): {topo.get('bridge_count', 0)}\n"
        response_text += f"- 🌿 Dead-End Nodes: {topo.get('dead_end_count', 0)}\n\n"
        
        # 压力模式
        pressure = features.get('pressure_patterns', {})
        if pressure:
            response_text += "**Pressure Patterns:**\n"
            response_text += f"- Mean Pressure: {pressure.get('mean', 0):.2f} m\n"
            response_text += f"- Pressure Range: {pressure.get('range', 0):.2f} m\n"
            response_text += f"- Uniformity (CV): {pressure.get('cv', 0):.3f}\n\n"
        
        # 流量模式
        flow = features.get('flow_patterns', {})
        if flow:
            response_text += "**Flow Patterns:**\n"
            response_text += f"- Mean Velocity: {flow.get('mean_velocity', 0):.3f} m/s\n"
            response_text += f"- Max Velocity: {flow.get('max_velocity', 0):.3f} m/s\n"
            response_text += f"- High-Flow Pipes: {flow.get('high_flow_pipe_count', 0)} (top 10%)\n\n"
        
        # 对称性指标
        symmetry = features.get('symmetry_metrics', {})
        if symmetry:
            balance_score = symmetry.get('flow_balance_score', 0)
            balance_emoji = "✅" if balance_score > 0.6 else "⚠️" if balance_score > 0.4 else "❌"
            response_text += "**Flow Balance:**\n"
            response_text += f"- Balance Score: {balance_score:.3f} {balance_emoji}\n"
            response_text += f"- Interpretation: {'Well balanced' if balance_score > 0.6 else 'Moderately balanced' if balance_score > 0.4 else 'Unbalanced flow distribution'}\n\n"
        
        # VLM 分析提示词（适用于高级用户）
        response_text += "---\n\n"
        response_text += "### 🤖 VLM Analysis Ready\n"
        response_text += "The generated heatmaps can be analyzed by Vision-Language Models (GPT-4o, Gemini-2.0-Pro) "
        response_text += "for advanced pattern recognition. Use the prompt template below:\n\n"
        response_text += "```\n"
        response_text += result['vlm_prompt'][:500] + "...\n"
        response_text += "```\n"
        
        return json.dumps({
            "status": "success",
            "msg": response_text,
            "raw_data": {
                "heatmap_paths": result['heatmap_paths'],
                "visual_features": result['visual_features'],
                "network_name": result['network_name']
            }
        }, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)})

@tool
def sensor_placer(inp_file: str, num_partitions: Optional[int] = None):
    """
    传感器布置工具 - 自动确定最佳传感器位置。
    当用户要求“布置传感器”、“传感器布置”、“监测点”、“传感器优化”或“传感器位置”时，请使用此工具。
    
    重要：该工具基于压力敏感性分析自动计算传感器的最佳数量和位置。
    您无需询问用户传感器计数或任何其他参数——只需使用 inp_file 调用此工具即可。
    
    该工具使用压力扰动分析来寻找具有最大检测覆盖范围的节点。
    每个分区的传感器计数是根据分区大小自动计算的（通常每个区域 2-10 个）。
    
    要求：必须先使用 'network_partitioner' 对网管进行分区。
    如果不存在分区结果，此工具将返回错误，要求用户先对管网进行分区。
    
    num_partitions：可选。指定用于传感器布置的分区计数。
    如果未指定，将根据现有的分区结果自动选择。
    """
    try:
        from optimization_utils.sensor_manager import run_sensor_placement_for_agent
        
        result = run_sensor_placement_for_agent(inp_file, num_partitions)
        
        # 处理不存在分区的情况
        if result["status"] == "no_partition":
            return json.dumps({
                "status": "error",
                "error": result["error"],
                "suggestion": "Please use network_partitioner to partition the network first, then run sensor placement."
            }, ensure_ascii=False)
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
        
        # 使用增强的 Markdown 格式化成功响应
        base_url = "http://127.0.0.1:5000"
        summary = result['summary']
        
        response_text = f"## ✅ {result['msg']}\n\n"
        response_text += "---\n\n"
        
        # 概览部分
        response_text += "### 📊 Placement Overview\n\n"
        response_text += f"| Metric | Value |\n"
        response_text += f"|--------|-------|\n"
        response_text += f"| **Total Sensors** | {summary['total_sensors']} |\n"
        response_text += f"| **Partitions** | {summary['num_partitions']} |\n"
        response_text += f"| **Sensitivity Threshold** | {summary['threshold']} |\n"
        response_text += f"| **Optimization Score** | {summary['score']:.4f} |\n\n"
        
        # 分区详情
        response_text += "### 📈 Partition Details\n\n"
        response_text += "| Partition | Sensors | Resilience | Coverage | Sensor Nodes |\n"
        response_text += "|-----------|---------|------------|----------|---------------|\n"
        
        for pid, details in summary['partition_details'].items():
            nodes_str = ", ".join(details['sensor_nodes'][:5])
            if len(details['sensor_nodes']) > 5:
                nodes_str += f" +{len(details['sensor_nodes'])-5}..."
            coverage = details.get('full_coverage_rate', 1.0) * 100
            response_text += f"| {pid} | {details['count']} | {details['resilience']:.4f} | {coverage:.1f}% | {nodes_str} |\n"
        
        response_text += "\n---\n\n"
        
        # 输出文件
        response_text += "### 📁 Output Files\n\n"
        
        # 可视化
        viz_filename = os.path.basename(result['viz_file'])
        response_text += f"**Visualization:**\n\n"
        response_text += f"![Sensor Placement]({base_url}/sensor_results/{viz_filename})\n\n"
        response_text += f"🖼️ [Download Visualization]({base_url}/sensor_results/{viz_filename})\n\n"
        
        # CSV 文件
        csv_filename = os.path.basename(result['sensor_file'])
        response_text += f"📄 [Download Sensor Placement CSV]({base_url}/sensor_results/{csv_filename})\n\n"
        
        # 建议
        response_text += "---\n\n"
        response_text += "### 🔧 Recommendations\n\n"
        response_text += "1. **Installation Location** - Install pressure sensors at recommended nodes\n"
        response_text += "2. **Resilience Score** - Indicates detection capability when some sensors fail\n"
        response_text += "3. **Priority** - Prioritize sensors with higher coverage rates\n"

        
        return json.dumps({
            "status": "success",
            "msg": response_text,
            "raw_data": result
        }, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)})

@tool
def leak_detector_trainer(inp_file: str, num_partitions: Optional[int] = None, n_scenarios: int = 3000):
    """
    Leak detection model training tool.
    Use when user asks to "train leak detection", "train LTFM model", "train anomaly detection",
    "训练漏损检测", "训练泄漏模型", or "开始漏损训练".

    Prerequisites: The network must have been partitioned (network_partitioner) AND
    sensors must have been placed (sensor_placer) before training.
    If prerequisites are not met, this tool will return an error with instructions.

    Parameters:
    - inp_file: EPANET INP file path
    - num_partitions: Optional partition count (auto-detected if not specified)
    - n_scenarios: Number of training scenarios (default: 3000, adjustable by user)
    """
    try:
        import glob
        import threading

        # --- Prerequisite check ---
        # Check partition results
        partition_files = glob.glob('partition_results/*partition_summary.json')
        if not partition_files:
            return json.dumps({
                "status": "error",
                "error": "No partition results found. Please run network_partitioner first.",
                "suggestion": "Use network_partitioner to partition the network, then sensor_placer to place sensors, before training."
            }, ensure_ascii=False)

        # Check sensor results
        sensor_files = glob.glob('sensor_results/sensor_placement_*.csv')
        if not sensor_files:
            return json.dumps({
                "status": "error",
                "error": "No sensor placement results found. Please run sensor_placer first.",
                "suggestion": "Use sensor_placer to place sensors before training the leak detection model."
            }, ensure_ascii=False)

        # Find best partition file (prefer FCM)
        fcm_files = [f for f in partition_files if 'fcm' in f.lower()]
        partition_file = fcm_files[0] if fcm_files else partition_files[0]

        # Auto-detect num_partitions if not specified
        if num_partitions is None:
            with open(partition_file, 'r', encoding='utf-8') as f:
                pdata = json.load(f)
            available_keys = [int(k) for k in pdata.keys() if k.isdigit()]
            num_partitions = max(available_keys) if available_keys else 5

        # --- Run training ---
        from wds_leak_main import load_config, setup_logging, train_mode
        import argparse

        config = load_config()
        setup_logging(config)
        os.makedirs(config['data']['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['data']['output_dir'], 'checkpoints'), exist_ok=True)

        # Create args namespace
        args = argparse.Namespace(
            inp=inp_file,
            partition=partition_file,
            num_partitions=num_partitions,
            n_scenarios=n_scenarios,
            skip_stage1=False,
            config=None
        )

        print(f"[LeakDetector] Starting training: inp={inp_file}, partition={partition_file}, "
              f"k={num_partitions}, scenarios={n_scenarios}")

        success = train_mode(config, args)

        if not success:
            return json.dumps({
                "status": "error",
                "error": "Leak detection model training failed. Check logs for details."
            })

        # --- Format success response ---
        base_url = "http://127.0.0.1:5000"
        output_dir = config['data']['output_dir']

        response_text = "## ✅ Leak Detection Model Training Complete\n\n"
        response_text += "---\n\n"
        response_text += "### 📊 Training Summary\n\n"
        response_text += f"| Parameter | Value |\n"
        response_text += f"|-----------|-------|\n"
        response_text += f"| **INP File** | {inp_file} |\n"
        response_text += f"| **Partition File** | {os.path.basename(partition_file)} |\n"
        response_text += f"| **Partitions** | {num_partitions} |\n"
        response_text += f"| **Training Scenarios** | {n_scenarios} |\n\n"

        response_text += "### 📁 Model Files\n\n"
        response_text += f"📥 [Download LTFM Model (best_model.pth)]({base_url}/leak_detection_output/checkpoints/best_model.pth)\n\n"
        response_text += f"📥 [Download NodeLocalizer Model (best_node_localizer.pth)]({base_url}/leak_detection_output/checkpoints/best_node_localizer.pth)\n\n"
        response_text += f"📥 [Download Graph2Vec Model (graph2vec_model.pth)]({base_url}/leak_detection_output/graph2vec_model.pth)\n\n"

        response_text += "---\n\n"
        response_text += "### 🔧 Next Steps\n\n"
        response_text += "1. **Run Inference** - Upload a CSV file with pressure data and ask to predict leaks\n"
        response_text += "2. **Download Template** - Use the inference template CSV as a reference for data format\n"
        response_text += f"3. **Template Download** - 📄 [Download Inference CSV Template]({base_url}/leak_detection_output/inference_template.csv)\n"

        return json.dumps({
            "status": "success",
            "msg": response_text
        }, ensure_ascii=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)})


@tool
def leak_detector_predictor(inp_file: str, csv_file: str, num_partitions: Optional[int] = None):
    """
    Leak detection inference tool - predict anomalies from pressure data.
    Use when user asks to "predict leaks", "detect leaks", "run inference",
    "漏损检测", "泄漏预测", "推理", or "anomaly detection".

    Prerequisites: A trained LTFM model must exist (run leak_detector_trainer first).

    Parameters:
    - inp_file: EPANET INP file path
    - csv_file: Path to CSV file containing pressure measurement data.
                Format: rows=timesteps, columns=node names, values=pressure (m).
    - num_partitions: Optional partition count (auto-detected if not specified)
    """
    try:
        import glob

        # --- Check trained model exists ---
        model_path = os.path.join('leak_detection_output', 'checkpoints', 'best_model.pth')
        if not os.path.exists(model_path):
            return json.dumps({
                "status": "error",
                "error": "No trained LTFM model found. Please run leak_detector_trainer first.",
                "suggestion": "Train the model first: 'Train leak detection model for <inp_file>'"
            }, ensure_ascii=False)

        # --- Check CSV file exists ---
        if not os.path.exists(csv_file):
            return json.dumps({
                "status": "error",
                "error": f"CSV file not found: {csv_file}",
                "suggestion": "Please upload a CSV file with pressure data first."
            })

        # Find partition file
        partition_files = glob.glob('partition_results/*partition_summary.json')
        if not partition_files:
            return json.dumps({
                "status": "error",
                "error": "No partition results found."
            })

        fcm_files = [f for f in partition_files if 'fcm' in f.lower()]
        partition_file = fcm_files[0] if fcm_files else partition_files[0]

        # Auto-detect num_partitions
        if num_partitions is None:
            with open(partition_file, 'r', encoding='utf-8') as f:
                pdata = json.load(f)
            available_keys = [int(k) for k in pdata.keys() if k.isdigit()]
            num_partitions = max(available_keys) if available_keys else 5

        # Find sensor file (optional)
        sensor_files = glob.glob('sensor_results/sensor_placement_*.csv')
        sensor_file = sensor_files[-1] if sensor_files else None

        # --- Run inference ---
        from wds_leak_main import load_config, setup_logging, inference_mode
        import argparse

        config = load_config()
        setup_logging(config)

        args = argparse.Namespace(
            inp=inp_file,
            partition=partition_file,
            num_partitions=num_partitions,
            sensor=sensor_file,
            graph2vec_model=None,
            ltfm_checkpoint=None,
            test_data=csv_file,
            output=os.path.join(config['data']['output_dir'], 'prediction_results.csv'),
            config=None
        )

        print(f"[LeakDetector] Starting inference: inp={inp_file}, csv={csv_file}, k={num_partitions}")

        success = inference_mode(config, args)

        if not success:
            return json.dumps({
                "status": "error",
                "error": "Leak detection inference failed. Check logs for details."
            })

        # --- Format success response ---
        base_url = "http://127.0.0.1:5000"
        output_dir = config['data']['output_dir']

        # Read prediction results if available
        result_path = os.path.join(output_dir, 'prediction_results.csv')
        result_summary = ""
        if os.path.exists(result_path):
            import pandas as pd
            df = pd.read_csv(result_path)
            result_summary = f"\n\n### 📋 Prediction Results\n\n"
            result_summary += f"| Metric | Value |\n"
            result_summary += f"|--------|-------|\n"
            for col in df.columns:
                val = df[col].iloc[0] if len(df) > 0 else "N/A"
                result_summary += f"| **{col}** | {val} |\n"

        response_text = "## 🔍 Leak Detection Inference Complete\n\n"
        response_text += "---\n\n"
        response_text += "### 📊 Input Summary\n\n"
        response_text += f"| Parameter | Value |\n"
        response_text += f"|-----------|-------|\n"
        response_text += f"| **INP File** | {inp_file} |\n"
        response_text += f"| **Pressure Data** | {os.path.basename(csv_file)} |\n"
        response_text += f"| **Partitions** | {num_partitions} |\n"
        if sensor_file:
            response_text += f"| **Sensor File** | {os.path.basename(sensor_file)} |\n"
        response_text += result_summary

        response_text += "\n---\n\n"
        response_text += "### 📁 Output Files\n\n"
        response_text += f"📄 [Download Prediction Results]({base_url}/leak_detection_output/prediction_results.csv)\n\n"

        return json.dumps({
            "status": "success",
            "msg": response_text
        }, ensure_ascii=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "error": str(e)})


tools = [hydraulic_inspector, reliability_assessor, graph_rag_retriever, network_partitioner, boundary_analyzer, zone_optimizer, visual_analyzer, sensor_placer, leak_detector_trainer, leak_detector_predictor]
tools_map = {
    "hydraulic_inspector": hydraulic_inspector,
    "graph_rag_retriever": graph_rag_retriever,
    "reliability_assessor": reliability_assessor,
    "network_partitioner": network_partitioner,
    "boundary_analyzer": boundary_analyzer,
    "zone_optimizer": zone_optimizer,
    "visual_analyzer": visual_analyzer,
    "sensor_placer": sensor_placer,
    "leak_detector_trainer": leak_detector_trainer,
    "leak_detector_predictor": leak_detector_predictor
}

# --- 手动代理执行器（对导入错误具有鲁棒性） ---

class SimpleAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Redis 连接
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print(">>> Redis Connected for History Persistence")
        except Exception as e:
            print(f"!!! Redis Connection Failed: {e} - Falling back to Memory")
            self.redis_client = None
            self.memory_sessions = {}

        self.system_text = """You are MM-WDS, a Physics-Informed Multi-Agent System for Water Distribution.
            Strategy:
            - If asked about "Reliability", ALWAYS CALL 'reliability_assessor'.
            - If asked about network statistics (nodes, pipes), hydraulic values (pressure), or "summary", Use 'hydraulic_inspector'.
            - If asked about a SPECIFIC node/link (e.g. "J-10"), use 'graph_rag_retriever'.
            - If asked to "partition", "divide", or "zone" the network (DMA design), use 'network_partitioner'.
            - If asked about "boundary pipes", "cut edges", "isolation valves", or "边界管段", use 'boundary_analyzer'.
              This tool automatically finds partition results - DO NOT ask user for partition_summary.json.
            - If asked to "optimize", "improve partition", or "find best valve config" for a specific zone count, use 'zone_optimizer'.
              Explain that this optimizes boundary valve status (open/closed) using NSGA-II.
            - If asked to "visualize", "show heatmap", "analyze pressure distribution", "show flow patterns", 
              "visual analysis", "生成热图", or "可视化分析", use 'visual_analyzer'.
            - If asked to "place sensors", "sensor placement", "monitoring points", "布置传感器", "监测点",
              "传感器优化", "传感器位置", or "布置监测点", use 'sensor_placer'.
            - If asked to "train leak detection", "train LTFM", "train anomaly detection", "训练漏损检测",
              "训练泄漏模型", or "开始漏损训练", use 'leak_detector_trainer'.
              This tool checks prerequisites (partition + sensor placement) automatically.
              Default training scenarios: 3000. User can adjust via natural language (e.g. "use 1000 scenarios").
              Pass the user-specified n_scenarios parameter if mentioned.
            - If asked to "predict leaks", "detect leaks", "run inference", "漏损检测", "泄漏预测",
              "推理", or "anomaly detection" with uploaded CSV data, use 'leak_detector_predictor'.
              The user must first upload a CSV file with pressure data. Pass the CSV file path.
            - Provide professional engineering diagnosis based on tool outputs.
            - The 'hydraulic_inspector' returns a JSON with 'statistics' and 'hydraulics'. READ IT CAREFULLY.
            
            IMPORTANT for Partitioning - Algorithm Selection:
            1. DEFAULT (Louvain Algorithm):
               - Use when user simply says "partition", "divide", or "zone" without specifying method.
            2. FCM Algorithm (Fuzzy C-Means):
               - Use when user EXPLICITLY mentions "FCM", "fuzzy", "sensitivity-based".
               - Pass algorithm="fcm" to the network_partitioner tool.
            
            IMPORTANT for Leak Detection:
            - Training requires: partition results + sensor results. If missing, instruct user.
            - Inference requires: trained model + CSV pressure data. CSV must have node names as columns.
            - When user mentions scenario count (e.g. "500 scenarios", "用500个场景"), pass as n_scenarios.
            
            CRITICAL - Output Formatting:
            - When a tool returns markdown links like [Download Image](http://...), you MUST include them
              EXACTLY as provided in your response. Do NOT convert them to plain text URLs.
            - This ensures users see clickable hyperlinks instead of raw URLs.
            """

        self.system_prompt = SystemMessage(content=self.system_text)

    def _serialize_msg(self, msg) -> str:
        data = {"type": msg.type, "content": msg.content}
        if isinstance(msg, ToolMessage):
            data["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            data["tool_calls"] = msg.tool_calls
        if msg.additional_kwargs:
            data["additional_kwargs"] = msg.additional_kwargs
        return json.dumps(data)

    def _deserialize_msg(self, data_str: str):
        try:
            data = json.loads(data_str)
            msg_type = data.get("type")
            content = data.get("content", "")
            
            if msg_type == "human": return HumanMessage(content=content)
            if msg_type == "ai": 
                msg = AIMessage(content=content)
                if "tool_calls" in data: msg.tool_calls = data["tool_calls"]
                if "additional_kwargs" in data: msg.additional_kwargs = data["additional_kwargs"]
                return msg
            if msg_type == "tool": return ToolMessage(tool_call_id=data.get("tool_call_id"), content=content)
            if msg_type == "system": return SystemMessage(content=content)
            return HumanMessage(content=content) # 备用方案
        except:
            return HumanMessage(content="")

    def get_session_history(self, session_id: str) -> List:
        history = []
        if self.redis_client:
            redis_key = f"chat:{session_id}"
            # 从 Redis 加载
            raw_msgs = self.redis_client.lrange(redis_key, 0, -1)
            if not raw_msgs:
                # 使用系统提示词初始化
                self.save_message(session_id, self.system_prompt)
                history = [self.system_prompt]
            else:
                history = [self._deserialize_msg(m) for m in raw_msgs]
        else:
            # 备用到内存
            if session_id not in self.memory_sessions:
                self.memory_sessions[session_id] = [self.system_prompt]
            history = self.memory_sessions[session_id]
        
        # 自愈：检查损坏的工具调用序列
        # 如果最后一条消息是带有 tool_calls 的 AIMessage，但后面没有 ToolMessage，则追加一个虚拟错误。
        if history and isinstance(history[-1], AIMessage) and history[-1].tool_calls:
            print(f"[{session_id}] Detected broken tool call sequence. Auto-fixing...")
            for tool_call in history[-1].tool_calls:
                dummy_tool_msg = ToolMessage(
                    tool_call_id=tool_call['id'],
                    content="Error: Tool execution interrupted or failed to save output. Please retry."
                )
                history.append(dummy_tool_msg)
                self.save_message(session_id, dummy_tool_msg)
                
        return history

    def summarize_conversation(self, messages: List[Any]) -> str:
        """根据消息生成对话的简短标题。"""
        try:
            # 从最后几条消息中提取文本进行摘要
            text_context = ""
            for msg in messages[:6]:
                # 跳过系统消息（SystemMessage）
                if isinstance(msg, SystemMessage):
                    continue
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                text_context += f"{role}: {msg.content}\n"
            
            prompt = f"""请根据以下对话内容，生成一个简短的标题（5-10个字，不要使用引号，直接返回标题文本）。
            
            对话内容：
            {text_context}
            
            标题："""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            title = response.content.strip().replace('"', '').replace("'", "")
            print(f"Generated title: {title}")
            return title
        except Exception as e:
            print(f"Title generation error: {e}")
            return "New Conversation"

    def save_message(self, session_id: str, msg):
        if self.redis_client:
            redis_key = f"chat:{session_id}"
            self.redis_client.rpush(redis_key, self._serialize_msg(msg))
            # 可选：7 天后过期
            self.redis_client.expire(redis_key, 60*60*24*7)
        else:
            self.memory_sessions[session_id].append(msg)

    def invoke(self, input_dict: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        session_id = "default"
        if config and "configurable" in config:
            session_id = config["configurable"].get("session_id", "default")
            
        history = self.get_session_history(session_id)
        
        user_msg = HumanMessage(content=input_dict["input"])
        history.append(user_msg) 
        self.save_message(session_id, user_msg)
        
        max_turns = 5
        turn = 0
        
        try:
            while turn < max_turns:
                turn += 1
                # 1. 调用大模型（LLM）
                response = self.llm_with_tools.invoke(history)
                history.append(response)
                self.save_message(session_id, response)

                # 2. 检查工具调用
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_call_id = tool_call["id"]
                        func = tools_map.get(tool_name)
                        
                        try:
                            if func:
                                print(f"[{session_id}] Executing tool: {tool_name} with {tool_args}")
                                tool_output = func.invoke(tool_args)
                                tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(tool_output))
                            else:
                                print(f"[{session_id}] Warning: Tool '{tool_name}' not found.")
                                tool_msg = ToolMessage(tool_call_id=tool_call_id, content=f"Error: Tool '{tool_name}' not available.")
                        except Exception as e:
                            print(f"[{session_id}] Error executing tool {tool_name}: {e}")
                            tool_msg = ToolMessage(tool_call_id=tool_call_id, content=f"Error executing tool: {str(e)}")
                        
                        history.append(tool_msg)
                        self.save_message(session_id, tool_msg)
                    
                    # 循环继续到下一次迭代，让大模型处理工具输出
                    continue
                else:
                    # 没有工具调用，这是最终回答
                    return {"output": response.content}
            
            return {"output": "Max turns reached. Please refine your query."}

        except Exception as e:
            print(f"Agent Loop Error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": f"System Error: {str(e)}. Please reset conversation."}

# 全局实例
chain_with_history = SimpleAgent()

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(">>> MM-WDS Agent Initialized (Manual Mode).")
    print(">>> Testing...")
    import time
    session_id = f"manual_test_{int(time.time())}"
    print(f">>> Usage Session ID: {session_id}")
    # 请求使用 FCM 分区进行传感器布置（自动传感器计数）
    res = chain_with_history.invoke(
        {"input": "Place sensors for dataset/Exa7.inp. Use the existing FCM partition with 5 zones."},
        config={"configurable": {"session_id": session_id}}
    )
    print(res)
