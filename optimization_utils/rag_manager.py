import os
import hashlib
import json
import wntr
import numpy as np
import pandas as pd

# Define storage directory
RAG_STORAGE_DIR = os.path.join(os.getcwd(), "rag_storage")
os.makedirs(RAG_STORAGE_DIR, exist_ok=True)

def compute_md5(file_path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cached_path(md5_hash):
    """Return path of cached JSON file based on given MD5 hash."""
    return os.path.join(RAG_STORAGE_DIR, f"{md5_hash}.json")

def ingest_inp_file(inp_file_path):
    """
    Ingest INP file:
    1. Compute MD5.
    2. Check if cache exists.
    3. If no cache, run WNTR simulation.
    4. Extract structural and hydraulic data.
    5. Save to JSON file.
    Returns metadata dictionary.
    """
    md5_hash = compute_md5(inp_file_path)
    cache_path = get_cached_path(md5_hash)
    
    if os.path.exists(cache_path):
        print(f">>> Loading cached GraphRAG data for {md5_hash}")
        with open(cache_path, "r", encoding='utf-8') as f:
            return json.load(f)

    print(f">>> Ingesting new network: {inp_file_path} (MD5: {md5_hash})")
    
    try:
        # Load and simulate
        wn = wntr.network.WaterNetworkModel(inp_file_path)
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        
        # Build NetworkX graph for topological queries
        G = wn.get_graph()
        
        # Extract basic statistics
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
        
        # Extract hydraulics summary (averages)
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
        
        # --- Detailed Element Extraction ---
        detailed_nodes = {}
        for node_name, node in wn.nodes():
            # Static properties
            node_data = {
                "type": node.node_type,
                "coordinates": node.coordinates if hasattr(node, "coordinates") else None,
                "elevation": float(node.elevation) if hasattr(node, "elevation") else 0.0,
                "base_demand": float(node.base_demand) if hasattr(node, "base_demand") else 0.0
            }
            
            # Dynamic results (averaged over time)
            if hasattr(results, "node"):
                if "pressure" in results.node:
                     node_data["pressure_avg"] = float(results.node['pressure'][node_name].mean())
                if "head" in results.node:
                     node_data["head_avg"] = float(results.node['head'][node_name].mean())
                if "demand" in results.node:
                     node_data["demand_avg"] = float(results.node['demand'][node_name].mean())

            # Topology
            if node_name in G:
                neighbors = list(G.neighbors(node_name))
                links = []
                # Find connected links
                for nbr in neighbors:
                    # MultiGraph might have multiple edges
                    edge_data = G.get_edge_data(node_name, nbr)
                    if edge_data:
                        for k, v in edge_data.items():
                             links.append(k) # k is the link name in wntr graph (link_name)
                
                node_data["neighbors"] = neighbors
                node_data["connected_links"] = list(set(links))

            detailed_nodes[node_name] = node_data

        detailed_links = {}
        for link_name, link in wn.links():
            # Static properties
            link_data = {
                "type": link.link_type,
                "start_node": link.start_node_name,
                "end_node": link.end_node_name,
            }
            if link.link_type == "Pipe":
                link_data["length"] = float(link.length)
                link_data["diameter"] = float(link.diameter)
                link_data["roughness"] = float(link.roughness)
            
            # Dynamic results
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
        # Save to disk
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
    Retrieve knowledge of the file.
    If not exists, trigger ingestion logic.
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
