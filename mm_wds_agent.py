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
    Hydraulic Inspector Tool.
    Use this tool when users ask about hydraulic properties, node counts, pipe counts, or simulation results.
    Do not run simulations blindly. This tool first checks the knowledge graph.
    """
    try:
        # Ingest/Retrieve data
        data = retrieve_knowledge(inp_file, query_type="summary")
        if "error" in data:
            return json.dumps({"status": "error", "error": data["error"]})

        # Return structured summary directly to LLM
        # The LLM will parse the JSON to answer user questions (e.g., "Calculate node count").
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
    Reliability Assessor Tool.
    Calculates Flow Entropy (FEF) and Network Resilience (NR).
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
    Topological Semantic GraphRAG Tool.
    Retrieves specific details about nodes or links (e.g., 'J-10' or 'Pipe-1') from the knowledge graph.
    Use this tool when users ask about specific network elements.
    """
    try:
        # Pass query_type="entity" to get detailed data
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
    Network Partitioner Tool.
    Use this tool when users ask to "partition" or "divide" the network into zones/communities.
    
    Algorithm Selection:
    - Default: "louvain" (Modularity-based community detection)
    - Alternative: "fcm" (Fuzzy C-Means clustering based on pressure sensitivity)
    
    Use FCM when user explicitly mentions "FCM", "fuzzy", "fuzzy c-means", "sensitivity-based", or "pressure sensitivity".
    
    Important - How 'num_partitions' works:
    - Louvain: Produces discrete partition counts; merges communities if target is lower.
    - FCM: Uses the specified cluster count directly.
    
    If user specifies a number (e.g., "5 zones", "divide into 3"), pass it as 'num_partitions'.
    Only pass algorithm="fcm" if user explicitly requests FCM-based partitioning.
    """
    try:
        # Default to Louvain algorithm
        use_fcm = algorithm and algorithm.lower() in ['fcm', 'fuzzy', 'fuzzy-c-means', 'fuzzycmeans']
        
        if use_fcm:
            # Use FCM partitioning
            from partition_utils.fcm_partition import run_fcm_partitioning_for_agent
            
            result = run_fcm_partitioning_for_agent(
                inp_file, 
                num_partitions=num_partitions or 5,
                fuzziness=1.5
            )
            
            if result["status"] == "error":
                return json.dumps({"status": "error", "error": result["error"]})
            
            # Format response for FCM
            base_url = "http://127.0.0.1:5000"
            
            response_text = f"## ✅ {result.get('msg', 'FCM partitioning completed.')}\n\n"
            response_text += "---\n\n"
            response_text += "### 📊 FCM Partitioning Results\n\n"
            
            # Partition stats table
            response_text += "| Partition | Node Count |\n"
            response_text += "|-----------|------------|\n"
            for partition, count in result['partition_stats'].items():
                response_text += f"| {partition} | {count} |\n"
            response_text += "\n"
            
            # Metrics
            response_text += "### 📈 Clustering Metrics\n\n"
            response_text += f"- **Fuzzy Partition Coefficient (FPC):** {result['metrics']['fpc']:.4f}\n"
            response_text += f"- **Convergence Iterations:** {result['metrics']['iterations']}\n"
            response_text += f"- **Fuzziness Parameter (m):** {result['fuzziness']}\n\n"
            
            # Visualization
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
            # Use Louvain algorithm (Default)
            from optimization_utils.partition_manager import run_partitioning_for_agent
            
            result = run_partitioning_for_agent(inp_file, target_k=num_partitions)
            
            if result["status"] == "error":
                 return json.dumps({"status": "error", "error": result["error"]})
            
            # Format user-friendly response with enhanced Markdown
            base_url = "http://127.0.0.1:5000"
            
            response_text = f"## ✅ {result.get('msg', 'Partitioning completed.')}\n\n"
            response_text += "---\n\n"
            response_text += "### 📊 Output Files\n\n"
            
            # Images
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
    Boundary Pipe Analyzer Tool.
    Use this tool when user asks about "boundary pipes", "cut edges", "isolation valves", 
    "connections between zones", or "boundary segments".
    
    This tool automatically finds partition results based on the INP file.
    No need to upload partition_summary.json separately.
    
    If num_partitions is not specified, it uses existing partition results.
    """
    try:
        from optimization_utils.zone_optimizer import analyze_boundary_pipes
        
        result = analyze_boundary_pipes(inp_file, target_k=num_partitions)
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
        
        # Format response with enhanced Markdown
        base_url = "http://127.0.0.1:5000"
        
        response_text = f"## 🔗 Boundary Pipe Analysis ({result['partition_count']} Zones)\n\n"
        response_text += f"**Total Boundary Pipes:** {result['boundary_pipe_count']}\n\n"
        response_text += "---\n\n"
        response_text += "### 📋 Boundary Pipe List\n\n"
        response_text += "| Pipe | From Node | To Node | Zone→Zone | Diameter (mm) | Length (m) |\n"
        response_text += "|------|-----------|---------|-----------|---------------|------------|\n"
        
        for p in result['boundary_pipes'][:20]:  # Limit to 20 for readability
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
    Run NSGA-II optimization for partition boundary configuration.
    Objectives: Maximize FEF, HRE, MRE, NR; Minimize Open Pipes.
    Use this tool when user asks to "optimize", "improve", or find "best configuration" 
    for a specific number of zones.
    
    This tool optimizes valve states (Open/Closed) for boundary pipes of a single partition count (e.g., 10 zones).
    It does not find the optimal number of partitions.
    """
    try:
        from optimization_utils.zone_optimizer import run_zone_optimization
        
        # Default to low iteration count for interactive speed unless user specifies otherwise
        result = run_zone_optimization(
            inp_file, 
            target_k=num_partitions,
            pop_size=pop_size, 
            n_gen=n_gen
        )
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
            
        # Enhanced Markdown output
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
    Visual Perception Analysis Tool.
    Use this tool when users ask to "visualize", "show heatmap", "analyze pressure distribution", 
    "show flow patterns", "visual analysis", or "generate network map".
    
    This tool generates visual heatmaps and extracts visual features from the network:
    - Pressure Heatmap (Blue=Low, Red=High)
    - Flow Visualization (Line width = Velocity)
    - Topological Anomalies (Dead-ends, Bridges)
    
    analysis_type options:
    - "pressure": Pressure heatmap only
    - "flow": Flow visualization only
    - "combined": Pressure and Flow (Default)
    - "features": Extract visual features without generating images
    """
    try:
        from partition_utils.visual_perception import (
            analyze_network_visually,
            extract_visual_features,
            get_vlm_prompt_template
        )
        import wntr
        
        base_url = "http://127.0.0.1:5000"
        
        # Run visual analysis
        result = analyze_network_visually(inp_file)
        
        if "error" in result:
            return json.dumps({"status": "error", "error": result["error"]})
        
        # Format response with enhanced markdown
        response_text = f"## 🎨 Visual Analysis Complete: {result['network_name']}\n\n"
        response_text += f"**Network Size:** {result['node_count']} nodes, {result['link_count']} links\n\n"
        response_text += "---\n\n"
        
        # Heatmap Images
        response_text += "### 📊 Generated Visualizations\n\n"
        
        for viz_type, path in result['heatmap_paths'].items():
            if analysis_type == "combined" or analysis_type == viz_type:
                filename = os.path.basename(path)
                label = viz_type.replace("_", " ").title()
                # Use relative path for web display
                rel_path = path.replace("\\", "/")
                response_text += f"**{label} Heatmap:**\n\n"
                response_text += f"![{label}]({base_url}/visual_outputs/{filename})\n\n"
                response_text += f"🖼️ [Download {label} Image]({base_url}/visual_outputs/{filename})\n\n"
        
        # Visual Features
        features = result['visual_features']
        response_text += "---\n\n"
        response_text += "### 🔍 Extracted Visual Features\n\n"
        
        # Topological Anomalies
        topo = features.get('topological_anomalies', {})
        response_text += "**Topological Analysis:**\n"
        response_text += f"- 🌉 Bridge Nodes (Critical): {topo.get('bridge_count', 0)}\n"
        response_text += f"- 🌿 Dead-End Nodes: {topo.get('dead_end_count', 0)}\n\n"
        
        # Pressure Patterns
        pressure = features.get('pressure_patterns', {})
        if pressure:
            response_text += "**Pressure Patterns:**\n"
            response_text += f"- Mean Pressure: {pressure.get('mean', 0):.2f} m\n"
            response_text += f"- Pressure Range: {pressure.get('range', 0):.2f} m\n"
            response_text += f"- Uniformity (CV): {pressure.get('cv', 0):.3f}\n\n"
        
        # Flow Patterns
        flow = features.get('flow_patterns', {})
        if flow:
            response_text += "**Flow Patterns:**\n"
            response_text += f"- Mean Velocity: {flow.get('mean_velocity', 0):.3f} m/s\n"
            response_text += f"- Max Velocity: {flow.get('max_velocity', 0):.3f} m/s\n"
            response_text += f"- High-Flow Pipes: {flow.get('high_flow_pipe_count', 0)} (top 10%)\n\n"
        
        # Symmetry Metrics
        symmetry = features.get('symmetry_metrics', {})
        if symmetry:
            balance_score = symmetry.get('flow_balance_score', 0)
            balance_emoji = "✅" if balance_score > 0.6 else "⚠️" if balance_score > 0.4 else "❌"
            response_text += "**Flow Balance:**\n"
            response_text += f"- Balance Score: {balance_score:.3f} {balance_emoji}\n"
            response_text += f"- Interpretation: {'Well balanced' if balance_score > 0.6 else 'Moderately balanced' if balance_score > 0.4 else 'Unbalanced flow distribution'}\n\n"
        
        # VLM Analysis Prompt (for power users)
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
    Sensor Placer Tool - Automatically determines optimal sensor locations.
    Use this tool when users ask to "place sensors", "sensor placement", "monitoring points", 
    "sensor optimization", or "sensor locations".
    
    IMPORTANT: This tool automatically calculates the optimal number and location of sensors 
    based on pressure sensitivity analysis.
    You do NOT need to ask the user for sensor counts or any other parameters - just call this tool with the inp_file.
    
    The tool uses pressure perturbation analysis to find nodes with maximum detection coverage.
    Sensor count per partition is automatically calculated based on partition size (typically 2-10 per zone).
    
    Requirement: The network MUST be partitioned using 'network_partitioner' first.
    If no partition results exist, this tool will return an error asking the user to partition the network first.
    
    num_partitions: Optional. Specify the partition count to use for sensor placement.
    If not specified, it will automatically select based on existing partition results.
    """
    try:
        from optimization_utils.sensor_manager import run_sensor_placement_for_agent
        
        result = run_sensor_placement_for_agent(inp_file, num_partitions)
        
        # Handle case where no partition exists
        if result["status"] == "no_partition":
            return json.dumps({
                "status": "error",
                "error": result["error"],
                "suggestion": "Please use network_partitioner to partition the network first, then run sensor placement."
            }, ensure_ascii=False)
        
        if result["status"] == "error":
            return json.dumps({"status": "error", "error": result["error"]})
        
        # Format success response with enhanced Markdown
        base_url = "http://127.0.0.1:5000"
        summary = result['summary']
        
        response_text = f"## ✅ {result['msg']}\n\n"
        response_text += "---\n\n"
        
        # Overview Section
        response_text += "### 📊 Placement Overview\n\n"
        response_text += f"| Metric | Value |\n"
        response_text += f"|--------|-------|\n"
        response_text += f"| **Total Sensors** | {summary['total_sensors']} |\n"
        response_text += f"| **Partitions** | {summary['num_partitions']} |\n"
        response_text += f"| **Sensitivity Threshold** | {summary['threshold']} |\n"
        response_text += f"| **Optimization Score** | {summary['score']:.4f} |\n\n"
        
        # Partition Details
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
        
        # Output Files
        response_text += "### 📁 Output Files\n\n"
        
        # Visualization
        viz_filename = os.path.basename(result['viz_file'])
        response_text += f"**Visualization:**\n\n"
        response_text += f"![Sensor Placement]({base_url}/sensor_results/{viz_filename})\n\n"
        response_text += f"🖼️ [Download Visualization]({base_url}/sensor_results/{viz_filename})\n\n"
        
        # CSV File
        csv_filename = os.path.basename(result['sensor_file'])
        response_text += f"📄 [Download Sensor Placement CSV]({base_url}/sensor_results/{csv_filename})\n\n"
        
        # Recommendations
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

# --- Manual Agent Executor (Robust to Import Errors) ---

class SimpleAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Redis Connection
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
            return HumanMessage(content=content) # Fallback
        except:
            return HumanMessage(content="")

    def get_session_history(self, session_id: str) -> List:
        history = []
        if self.redis_client:
            redis_key = f"chat:{session_id}"
            # Load from Redis
            raw_msgs = self.redis_client.lrange(redis_key, 0, -1)
            if not raw_msgs:
                # Initialize with system prompt
                self.save_message(session_id, self.system_prompt)
                history = [self.system_prompt]
            else:
                history = [self._deserialize_msg(m) for m in raw_msgs]
        else:
            # Fallback to Memory
            if session_id not in self.memory_sessions:
                self.memory_sessions[session_id] = [self.system_prompt]
            history = self.memory_sessions[session_id]
        
        # Self-healing: Check for broken tool call sequences
        # If the last message is an AIMessage with tool_calls but no following ToolMessage, append a dummy error.
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
        """Generate a short title based on conversation messages."""
        try:
            # Extract text from the last few messages for summarization
            text_context = ""
            for msg in messages[:6]:
                # Skip SystemMessage
                if isinstance(msg, SystemMessage):
                    continue
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                text_context += f"{role}: {msg.content}\n"
            
            prompt = f"""Please generate a short title (5-10 words, do not use quotes, return only the title text) based on the following conversation.
            
            Conversation:
            {text_context}
            
            Title:"""
            
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
            # Optional: Expire after 7 days
            self.redis_client.expire(redis_key, 60*60*24*7)
        else:
            if session_id not in self.memory_sessions:
                 self.memory_sessions[session_id] = []
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
                # 1. Call LLM
                response = self.llm_with_tools.invoke(history)
                history.append(response)
                self.save_message(session_id, response)

                # 2. Check Tool Calls
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
                    
                    # Continue loop to let LLM process tool outputs
                    continue
                else:
                    # No tool calls, this is the final answer
                    return {"output": response.content}
            
            return {"output": "Max turns reached. Please refine your query."}

        except Exception as e:
            print(f"Agent Loop Error: {e}")
            import traceback
            traceback.print_exc()
            return {"output": f"System Error: {str(e)}. Please reset conversation."}

# Global Instance
chain_with_history = SimpleAgent()

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(">>> MM-WDS Agent Initialized (Manual Mode).")
    print(">>> Testing...")
    import time
    session_id = f"manual_test_{int(time.time())}"
    print(f">>> Usage Session ID: {session_id}")
    # Request sensor placement with FCM (auto sensor count)
    res = chain_with_history.invoke(
        {"input": "Place sensors for dataset/Exa7.inp. Use the existing FCM partition with 5 zones."},
        config={"configurable": {"session_id": session_id}}
    )
    print(res)
