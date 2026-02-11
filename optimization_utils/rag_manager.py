import os
import hashlib
import json
import wntr
import numpy as np
import pandas as pd

# 定义存储目录
RAG_STORAGE_DIR = os.path.join(os.getcwd(), "rag_storage")
os.makedirs(RAG_STORAGE_DIR, exist_ok=True)

def compute_md5(file_path):
    """通过 MD5 算法计算文件的哈希值。"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cached_path(md5_hash):
    """根据给定的 MD5 哈希值，返回缓存的 JSON 文件的路径。"""
    return os.path.join(RAG_STORAGE_DIR, f"{md5_hash}.json")

def ingest_inp_file(inp_file_path):
    """
    摄取 INP 文件：
    1. 计算 MD5。
    2. 检查是否已有缓存。
    3. 若无缓存，则运行 WNTR 模拟。
    4. 提取结构和水力数据。
    5. 保存到 JSON 文件。
    返回元数据字典。
    """
    md5_hash = compute_md5(inp_file_path)
    cache_path = get_cached_path(md5_hash)
    
    if os.path.exists(cache_path):
        print(f">>> Loading cached GraphRAG data for {md5_hash}")
        with open(cache_path, "r", encoding='utf-8') as f:
            return json.load(f)

    print(f">>> Ingesting new network: {inp_file_path} (MD5: {md5_hash})")
    
    try:
        # 加载并模拟
        wn = wntr.network.WaterNetworkModel(inp_file_path)
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        
        # 构建用于拓扑查询的 NetworkX 图
        G = wn.get_graph()
        
        # 提取基础统计数据
        stats = {
            "node_count": wn.num_nodes,
            "link_count": wn.num_links,
            "reservoir_count": wn.num_reservoirs,
            "tank_count": wn.num_tanks,
            "pump_count": wn.num_pumps,
            "valve_count": wn.num_valves,
            "junction_count": wn.num_junctions,
            "pipe_count": wn.num_pipes
        }
        
        # 提取水力摘要（平均值）
        try:
            avg_pressure = results.node['pressure'].mean().mean()
            min_pressure = results.node['pressure'].min().min()
            max_pressure = results.node['pressure'].max().max()
        except:
            avg_pressure = 0
            min_pressure = 0
            max_pressure = 0

        hydraulics_summary = {
            "avg_pressure": float(avg_pressure),
            "min_pressure": float(min_pressure),
            "max_pressure": float(max_pressure)
        }
        
        # --- 详细元素提取 ---
        detailed_nodes = {}
        for node_name, node in wn.nodes():
            # 静态属性
            node_data = {
                "type": node.node_type,
                "coordinates": node.coordinates if hasattr(node, "coordinates") else None,
                "elevation": float(node.elevation) if hasattr(node, "elevation") else 0.0,
                "base_demand": float(node.base_demand) if hasattr(node, "base_demand") else 0.0
            }
            
            # 动态结果（随时间平均）
            if hasattr(results, "node"):
                if "pressure" in results.node:
                     node_data["pressure_avg"] = float(results.node['pressure'][node_name].mean())
                if "head" in results.node:
                     node_data["head_avg"] = float(results.node['head'][node_name].mean())
                if "demand" in results.node:
                     node_data["demand_avg"] = float(results.node['demand'][node_name].mean())

            # 拓扑结构
            if node_name in G:
                neighbors = list(G.neighbors(node_name))
                links = []
                # 查找连接的链路
                for nbr in neighbors:
                    # MultiGraph 可能具有多条边
                    edge_data = G.get_edge_data(node_name, nbr)
                    if edge_data:
                        for k, v in edge_data.items():
                             links.append(k) # k 是 wntr 图中的链路名称 (link_name)
                
                node_data["neighbors"] = neighbors
                node_data["connected_links"] = list(set(links))

            detailed_nodes[node_name] = node_data

        detailed_links = {}
        for link_name, link in wn.links():
            # 静态属性
            link_data = {
                "type": link.link_type,
                "start_node": link.start_node_name,
                "end_node": link.end_node_name,
            }
            if link.link_type == "Pipe":
                link_data["length"] = float(link.length)
                link_data["diameter"] = float(link.diameter)
                link_data["roughness"] = float(link.roughness)
            
            # 动态结果
            if hasattr(results, "link"):
                if "flowrate" in results.link:
                    link_data["flowrate_avg"] = float(results.link['flowrate'][link_name].mean())
                if "velocity" in results.link:
                    link_data["velocity_avg"] = float(results.link['velocity'][link_name].mean())

            detailed_links[link_name] = link_data

        # -----------------------------------
        
        node_ids = list(wn.node_name_list)
        link_ids = list(wn.link_name_list)
        
        data = {
            "md5": md5_hash,
            "filename": os.path.basename(inp_file_path),
            "stats": stats,
            "hydraulics": hydraulics_summary,
            "nodes": node_ids,
            "links": link_ids,
            "detailed_nodes": detailed_nodes,
            "detailed_links": detailed_links,
            "status": "ingested"
        }
        
        print(f">>> Ingestion successful. Stats: {stats}")
        # 保存到磁盘
        with open(cache_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return data
        
    except Exception as e:
        print(f"!!! Error ingesting {inp_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}

def retrieve_knowledge(inp_file_path, query_type="summary", entity_id=None):
    """
    检索文件的知识。
    如果不存在，则触发摄取逻辑。
    """
    data = ingest_inp_file(inp_file_path)
    if "error" in data:
        return data
        
    if query_type == "summary":
        return {
            "stats": data["stats"],
            "hydraulics": data["hydraulics"],
            "filename": data["filename"]
        }
    
    if query_type == "entity" and entity_id:
        if entity_id in data.get("detailed_nodes", {}):
             return {"type": "Node", "data": data["detailed_nodes"][entity_id]}
        if entity_id in data.get("detailed_links", {}):
             return {"type": "Link", "data": data["detailed_links"][entity_id]}
        return {"error": "Entity not found"}

    return data
