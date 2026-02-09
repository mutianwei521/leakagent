"""
HydroSim Hydraulic Simulation Agent
Responsible for processing .inp files, performing network analysis and hydraulic calculations
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from .base_agent import BaseAgent
from .intent_classifier_fast import FastIntentClassifier as IntentClassifier

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

class HydroSim(BaseAgent):
    """Hydraulic simulation agent"""
    
    def __init__(self):
        super().__init__("HydroSim")

        if not WNTR_AVAILABLE:
            self.log_error("WNTR library not installed, hydraulic calculation function unavailable")

        self.intent_classifier = IntentClassifier()
        self.downloads_folder = 'downloads'
        os.makedirs(self.downloads_folder, exist_ok=True)

        # Cache mechanism: avoid parsing same file repeatedly
        self._network_cache = {}  # {file_path: {network_info, last_modified}}
    
    def parse_network(self, inp_file_path: str):
        """Parse network file, extract basic information"""
        if not WNTR_AVAILABLE:
            return {'error': 'WNTR library not installed'}

        try:
            # Check cache
            if inp_file_path in self._network_cache:
                file_mtime = os.path.getmtime(inp_file_path)
                cached_data = self._network_cache[inp_file_path]
                if cached_data['last_modified'] == file_mtime:
                    self.log_info(f"Using cached network info: {inp_file_path}")
                    return cached_data['network_info']

            self.log_info(f"Start parsing network file: {inp_file_path}")

            # Read network file
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            
            # Extract key information
            network_info = {
                'nodes': {
                    'junctions': len(wn.junction_name_list),
                    'reservoirs': len(wn.reservoir_name_list),
                    'tanks': len(wn.tank_name_list),
                    'total': len(wn.node_name_list)
                },
                'links': {
                    'pipes': len(wn.pipe_name_list),
                    'pumps': len(wn.pump_name_list),
                    'valves': len(wn.valve_name_list),
                    'total': len(wn.link_name_list)
                },
                'network_stats': {
                    'total_length': float(sum([wn.get_link(pipe).length for pipe in wn.pipe_name_list])) if len(wn.pipe_name_list) > 0 else 0,
                    'simulation_duration': wn.options.time.duration,
                    'hydraulic_timestep': wn.options.time.hydraulic_timestep,
                    'pattern_timestep': wn.options.time.pattern_timestep
                }
            }
            
            # Add detailed topology info for visualization
            network_info['topology'] = self._extract_topology_data(wn)

            self.log_info(f"Network parsing complete: {network_info['nodes']['total']} nodes, {network_info['links']['total']} links")

            # Update cache
            file_mtime = os.path.getmtime(inp_file_path)
            self._network_cache[inp_file_path] = {
                'network_info': network_info,
                'last_modified': file_mtime
            }

            return network_info
            
        except Exception as e:
            error_msg = f"Failed to parse network file: {e}"
            self.log_error(error_msg)
            return {'error': error_msg}

    def _extract_topology_data(self, wn):
        """Extract topology data for visualization"""
        try:
            topology = {
                'nodes': [],
                'links': []
            }

            # Extract node information
            for node_name in wn.node_name_list:
                node = wn.get_node(node_name)

                # Determine node type
                node_type = 'junction'  # Default type
                class_name = type(node).__name__

                # Determine type based on WNTR class name
                if 'Reservoir' in class_name:
                    node_type = 'reservoir'
                elif 'Tank' in class_name:
                    node_type = 'tank'
                elif 'Junction' in class_name:
                    node_type = 'junction'
                else:
                    # Try other attributes
                    if hasattr(node, '_node_type'):
                        node_type = node._node_type.lower()
                    elif hasattr(node, 'node_type'):
                        node_type = node.node_type.lower()
                    else:
                        # Last resort fallback
                        class_lower = class_name.lower()
                        if 'reservoir' in class_lower:
                            node_type = 'reservoir'
                        elif 'tank' in class_lower:
                            node_type = 'tank'

                node_data = {
                    'id': node_name,
                    'type': node_type,
                    'coordinates': [node.coordinates[0], node.coordinates[1]] if hasattr(node, 'coordinates') and node.coordinates else [0, 0]
                }

                # Add node specific attributes
                if hasattr(node, 'elevation'):
                    node_data['elevation'] = float(node.elevation) if node.elevation is not None else 0.0
                if hasattr(node, 'base_demand'):
                    node_data['base_demand'] = float(node.base_demand) if node.base_demand is not None else 0.0
                if hasattr(node, 'head'):
                    node_data['head'] = float(node.head) if node.head is not None else 0.0
                if hasattr(node, 'init_level') and node.init_level is not None:
                    node_data['init_level'] = float(node.init_level)
                if hasattr(node, 'max_level') and node.max_level is not None:
                    node_data['max_level'] = float(node.max_level)
                if hasattr(node, 'min_level') and node.min_level is not None:
                    node_data['min_level'] = float(node.min_level)

                topology['nodes'].append(node_data)

            # Extract link information
            for link_name in wn.link_name_list:
                link = wn.get_link(link_name)

                # Determine link type
                link_type = 'pipe'  # Default type
                class_name = type(link).__name__

                # Determine type based on WNTR class name
                if 'Pump' in class_name:
                    link_type = 'pump'
                elif 'Valve' in class_name:
                    link_type = 'valve'
                elif 'Pipe' in class_name:
                    link_type = 'pipe'
                else:
                    # Try other attributes
                    if hasattr(link, '_link_type'):
                        link_type = link._link_type.lower()
                    elif hasattr(link, 'link_type'):
                        link_type = link.link_type.lower()
                    else:
                        # Last resort fallback
                        class_lower = class_name.lower()
                        if 'pump' in class_lower:
                            link_type = 'pump'
                        elif 'valve' in class_lower:
                            link_type = 'valve'

                link_data = {
                    'id': link_name,
                    'type': link_type,
                    'start_node': link.start_node_name,
                    'end_node': link.end_node_name
                }

                # Add link specific attributes
                if hasattr(link, 'length'):
                    link_data['length'] = float(link.length) if link.length is not None else 0.0
                if hasattr(link, 'diameter'):
                    link_data['diameter'] = float(link.diameter) if link.diameter is not None else 0.0
                if hasattr(link, 'roughness'):
                    link_data['roughness'] = float(link.roughness) if link.roughness is not None else 0.0
                if hasattr(link, 'minor_loss'):
                    link_data['minor_loss'] = float(link.minor_loss) if link.minor_loss is not None else 0.0

                topology['links'].append(link_data)

            return topology

        except Exception as e:
            self.log_error(f"Failed to extract topology data: {e}")
            return {'nodes': [], 'links': []}
    
    def run_hydraulic_simulation(self, inp_file_path: str):
        """Run hydraulic calculation"""
        if not WNTR_AVAILABLE:
            return {'success': False, 'error': 'WNTR library not installed'}
        
        try:
            self.log_info("Starting hydraulic calculation...")
            
            # Create network model
            wn = wntr.network.WaterNetworkModel(inp_file_path)
            
            # Run hydraulic calculation
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()
            
            # Extract key data
            simulation_data = {
                'node_pressure': results.node['pressure'],
                'node_demand': results.node['demand'],
                'link_flowrate': results.link['flowrate'],
                'link_velocity': results.link['velocity']
            }
            
            self.log_info("Hydraulic calculation complete")
            return {'success': True, 'data': simulation_data}
            
        except Exception as e:
            error_msg = f"Hydraulic calculation failed: {e}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def save_simulation_to_csv(self, simulation_data: dict, conversation_id: str):
        """Save hydraulic calculation results to CSV file"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hydraulic_simulation_{conversation_id[:8]}_{timestamp}.csv"
            file_path = os.path.join(self.downloads_folder, filename)

            # Prepare data
            all_data = []

            # ProcessNode pressure data
            if 'node_pressure' in simulation_data:
                pressure_df = simulation_data['node_pressure']
                # WNTR DataFrame structure: rows are time steps, columns are Node IDs
                for time_idx in pressure_df.index:  # Time steps in row index
                    for node_id in pressure_df.columns:  # Node IDs in column index
                        try:
                            # time_idx is time step (seconds), convert to hours
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # Default value
                        all_data.append({
                            'Time(hours)': time_hours,
                            'Node ID': str(node_id),  # Ensure string
                            'Data type': 'Node pressure',
                            'Value': pressure_df.loc[time_idx, node_id],
                            'Unit': 'm'
                        })
            
            # ProcessNode demand data
            if 'node_demand' in simulation_data:
                demand_df = simulation_data['node_demand']
                # WNTR DataFrame structure: rows are time steps, columns are Node IDs
                for time_idx in demand_df.index:  # Time steps in row index
                    for node_id in demand_df.columns:  # Node IDs in column index
                        try:
                            # time_idx is time step (seconds), convert to hours
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # Default value
                        all_data.append({
                            'Time(hours)': time_hours,
                            'Node ID': str(node_id),  # Ensure string
                            'Data type': 'Node demand',
                            'Value': demand_df.loc[time_idx, node_id],
                            'Unit': 'L/s'
                        })
            
            # ProcessPipe flow data
            if 'link_flowrate' in simulation_data:
                flow_df = simulation_data['link_flowrate']
                # WNTR DataFrame structure: rows are time steps, columns are Pipe IDs
                for time_idx in flow_df.index:  # Time steps in row index
                    for link_id in flow_df.columns:  # Pipe IDs in column index
                        try:
                            # time_idx is time step (seconds), convert to hours
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # Default value
                        all_data.append({
                            'Time(hours)': time_hours,
                            'Pipe ID': str(link_id),  # Ensure string
                            'Data type': 'Pipe flow',
                            'Value': flow_df.loc[time_idx, link_id],
                            'Unit': 'L/s'
                        })
            
            # ProcessPipe velocity data
            if 'link_velocity' in simulation_data:
                velocity_df = simulation_data['link_velocity']
                # WNTR DataFrame structure: rows are time steps, columns are Pipe IDs
                for time_idx in velocity_df.index:  # Time steps in row index
                    for link_id in velocity_df.columns:  # Pipe IDs in column index
                        try:
                            # time_idx is time step (seconds), convert to hours
                            time_hours = float(time_idx) / 3600
                        except (ValueError, TypeError):
                            time_hours = 0  # Default value
                        all_data.append({
                            'Time(hours)': time_hours,
                            'Pipe ID': str(link_id),  # Ensure string
                            'Data type': 'Pipe velocity',
                            'Value': velocity_df.loc[time_idx, link_id],
                            'Unit': 'm/s'
                        })
            
            # Save as CSV
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                file_size = os.path.getsize(file_path)
                self.log_info(f"CSV file saved successfully: {filename} ({file_size} bytes)")
                
                return {
                    'success': True,
                    'filename': filename,
                    'file_path': file_path,
                    'download_url': f'/download/{filename}',
                    'file_size': file_size,
                    'records_count': len(all_data)
                }
            else:
                return {'success': False, 'error': 'No data to save'}
                
        except Exception as e:
            error_msg = f"Failed to save CSV file: {e}"
            self.log_error(error_msg)
            return {'success': False, 'error': error_msg}

    def build_simulation_prompt(self, network_info: dict, simulation_data: dict, user_message: str, csv_info: dict):
        """Build hydraulic calculation analysis prompt with download link"""
        prompt = f"""
You are a professional water distribution network analysis expert. Now need to analyze the following network system: 

Network basic information: 
- Total nodes: {network_info['nodes']['total']} (Junctions: {network_info['nodes']['junctions']}, Reservoirs: {network_info['nodes']['reservoirs']}, Tanks: {network_info['nodes']['tanks']})
- Total links: {network_info['links']['total']} (Pipes: {network_info['links']['pipes']}, Pumps: {network_info['links']['pumps']}, Valves: {network_info['links']['valves']})
- Total network length: {network_info['network_stats']['total_length']:.2f} meters
- Simulation duration: {network_info['network_stats']['simulation_duration']} seconds

✅ Hydraulic calculation completed successfully!

Calculation results include: 
- Node pressure distribution data
- Node demand data
- Pipe flow data
- Pipe velocity data

📊 Detailed data saved as CSV file: {csv_info['filename']}
File size: {csv_info['file_size']} bytes, total {csv_info['records_count']} records

User question: {user_message}

Please provide professional analysis and recommendations based on the network information and hydraulic calculation results. 
Also inform the user that detailed calculation data can be downloaded for further analysis. 

Please use the following signature format at the end of your reply: 

Best regards, 

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_analysis_prompt(self, network_info: dict, user_message: str):
        """Build network structure analysis prompt"""
        prompt = f"""
You are a professional water distribution network analysis expert. Now need to analyze the structure of following network system: 

Network basic information: 
- Total nodes: {network_info['nodes']['total']} (Junctions: {network_info['nodes']['junctions']}, Reservoirs: {network_info['nodes']['reservoirs']}, Tanks: {network_info['nodes']['tanks']})
- Total links: {network_info['links']['total']} (Pipes: {network_info['links']['pipes']}, Pumps: {network_info['links']['pumps']}, Valves: {network_info['links']['valves']})
- Total network length: {network_info['network_stats']['total_length']:.2f} meters

User question: {user_message}

Please provide professional analysis and recommendations based on the network structure information. 
If user needs detailed hydraulic calculation data, please suggest running hydraulic calculation. 

Please use the following signature format at the end of your reply: 

Best regards, 

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_general_prompt(self, network_info: dict, user_message: str):
        """Build general consultation prompt"""
        prompt = f"""
You are a professional water distribution network analysis expert. User has uploaded a network file (.inp format). 

Network basic information: 
- Total nodes: {network_info['nodes']['total']}
- Total links: {network_info['links']['total']}
- Total network length: {network_info['network_stats']['total_length']:.2f} meters

User question: {user_message}

Please answer the user's question and introduce available analysis functions: 
1. Network structure analysis
2. Hydraulic calculation and simulation
3. Data export and download

Please use the following signature format at the end of your reply: 

Best regards, 

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def build_error_prompt(self, network_info: dict, user_message: str, error_message: str):
        """Build error handling prompt"""
        prompt = f"""
You are a professional water distribution network analysis expert. A problem was encountered while processing user request. 

Network basic information: 
- Total nodes: {network_info['nodes']['total']}
- Total links: {network_info['links']['total']}

User question: {user_message}

Problem encountered: {error_message}

Please explain the problem to the user and provide possible solutions or alternative suggestions. 

Please use the following signature format at the end of your reply: 

Best regards, 

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
        return prompt

    def process(self, inp_file_path: str, user_message: str, conversation_id: str):
        """Main method to process network file and user message"""
        self.log_info(f"Starting to process network file: {inp_file_path}")

        # Step 1: Parsing network file
        network_info = self.parse_network(inp_file_path)
        if 'error' in network_info:
            return {
                'success': False,
                'response': f"Failed to parse network file: {network_info['error']}",
                'network_info': None,
                'intent': 'error',
                'confidence': 0.0
            }

        # Step 2: Intelligent intent recognition
        intent_result = self.intent_classifier.classify_intent(user_message)
        intent = intent_result['intent']
        confidence = intent_result['confidence']

        self.log_info(f"Recognized intent: {intent}, Confidence: {confidence:.3f}")

        csv_info = None
        prompt = ""

        # Step 3: Execute different operations based on intent
        if intent == 'hydraulic_simulation' and confidence > 0.7:
            # Execute hydraulic calculation
            simulation_result = self.run_hydraulic_simulation(inp_file_path)

            if simulation_result['success']:
                # Save CSV file
                csv_info = self.save_simulation_to_csv(
                    simulation_result['data'],
                    conversation_id
                )

                if csv_info['success']:
                    prompt = self.build_simulation_prompt(
                        network_info,
                        simulation_result['data'],
                        user_message,
                        csv_info
                    )
                else:
                    prompt = self.build_error_prompt(
                        network_info,
                        user_message,
                        f"Hydraulic calculation succeeded, but failed to save CSV file: {csv_info['error']}"
                    )
            else:
                prompt = self.build_error_prompt(
                    network_info,
                    user_message,
                    f"Hydraulic calculation failed: {simulation_result['error']}"
                )

        elif intent == 'network_analysis' and confidence > 0.6:
            # Structure analysis
            prompt = self.build_analysis_prompt(network_info, user_message)

        else:
            # General consultation
            prompt = self.build_general_prompt(network_info, user_message)

        return {
            'success': True,
            'prompt': prompt,
            'csv_info': csv_info,
            'network_info': network_info,
            'intent': intent,
            'confidence': confidence
        }
