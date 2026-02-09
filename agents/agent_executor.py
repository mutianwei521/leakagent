"""
Agent Executor
Responsible for coordinating execution of different agents based on LLM analysis results
"""
import os
import json
import openai
from datetime import datetime
from .base_agent import BaseAgent

class AgentExecutor(BaseAgent):
    """Agent executor"""
    
    def __init__(self, hydro_sim, partition_sim, sensor_placement, leak_detection):
        super().__init__("AgentExecutor")
        
        # Register agents
        self.agents = {
            "Network analysis": hydro_sim,
            "Network partition": partition_sim,
            "Sensor placement": sensor_placement,
            "Leak model training": leak_detection,
            "Leak detection": leak_detection,
            "Hydraulic simulation": hydro_sim,
            "Topology analysis": hydro_sim
        }
        
        # Set OpenAI API configuration
        openai.api_base = ""
        openai.api_key = ""
    
    def execute_step(self, step_name: str, step_info: dict, conversation_id: str, user_message: str):
        """Execute single step"""
        try:
            self.log_info(f"Starting execution step: {step_name}")
            
            # Get corresponding agent
            agent = self.agents.get(step_name)
            if not agent:
                return {
                    'success': False,
                    'error': f'Agent not found: {step_name}',
                    'step_name': step_name
                }
            
            # Execute different logic based on step type
            if step_name == "Network analysis":
                return self._execute_network_analysis(agent, step_info, conversation_id, user_message)
            elif step_name == "Network partition":
                return self._execute_partition_analysis(agent, step_info, conversation_id, user_message)
            elif step_name == "Sensor placement":
                return self._execute_sensor_placement(agent, step_info, conversation_id, user_message)
            elif step_name == "Leak model training":
                return self._execute_leak_training(agent, step_info, conversation_id, user_message)
            elif step_name == "Leak detection":
                return self._execute_leak_detection(agent, step_info, conversation_id, user_message)
            else:
                return self._execute_generic_step(agent, step_info, conversation_id, user_message)
                
        except Exception as e:
            self.log_error(f"Execution step failed {step_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': step_name
            }
    
    def _execute_network_analysis(self, agent, step_info, conversation_id, user_message):
        """Execute network analysis"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': 'Missing INP file',
                'step_name': 'Network analysis'
            }
        
        # Call hydraulic simulation agent
        result = agent.process(inp_file, user_message, conversation_id)
        
        if result['success']:
            self.log_info("Network analysis execution success")
            return {
                'success': True,
                'step_name': 'Network analysis',
                'result': result,
                'agent_type': 'hydro_sim'
            }
        else:
            return {
                'success': False,
                'error': result.get('response', 'Network analysis failed'),
                'step_name': 'Network analysis'
            }
    
    def _execute_partition_analysis(self, agent, step_info, conversation_id, user_message):
        """Execute network partition"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': 'Missing INP file',
                'step_name': 'Network partition'
            }
        
        # Build partition request message
        partition_message = user_message
        if "Partition" not in user_message:
            partition_message = f"{user_message} Please perform network partition analysis"
        
        # Call partition agent
        result = agent.process(inp_file, partition_message, conversation_id)
        
        if result['success']:
            self.log_info("Network partition execution success")
            return {
                'success': True,
                'step_name': 'Network partition',
                'result': result,
                'agent_type': 'partition_sim',
                'csv_file': result.get('csv_info', {}).get('filepath')
            }
        else:
            return {
                'success': False,
                'error': result.get('response', 'Network partition failed'),
                'step_name': 'Network partition'
            }
    
    def _execute_sensor_placement(self, agent, step_info, conversation_id, user_message):
        """Execute sensor placement"""
        try:
            inp_file = step_info.get('input_file')
            # Prefer using partition file specified in execution plan, otherwise find existing files
            partition_csv = step_info.get('partition_file') or self._find_partition_csv(conversation_id)

            self.log_info(f"Sensor placement parameter check: inp_file={inp_file}, partition_csv={partition_csv}")

            # Check if this is incremental execution based on existing partition
            if step_info.get('partition_file'):
                self.log_info(f"Using specified partition file: {partition_csv}")
            elif partition_csv:
                self.log_info(f"Using detected partition file: {partition_csv}")

            if not inp_file:
                return {
                    'success': False,
                    'error': 'Missing INP file',
                    'step_name': 'Sensor placement'
                }

            if not partition_csv:
                return {
                    'success': False,
                    'error': 'Missing partition CSV file, please perform network partition first',
                    'step_name': 'Sensor placement'
                }

            # Build sensor placement request message
            sensor_message = user_message
            if "sensor" not in user_message.lower() and "monitor" not in user_message.lower():
                sensor_message = f"{user_message} Please perform sensor placement optimization"

            self.log_info(f"Starting to call sensor placement agent: {sensor_message}")

            # Call sensor placement agent
            result = agent.process(inp_file, partition_csv, sensor_message, conversation_id)

            self.log_info(f"Sensor placement agent call complete, result: {result.get('success', False)}")

            if result['success']:
                self.log_info("Sensor placement execution success")
                return {
                    'success': True,
                    'step_name': 'Sensor placement',
                    'result': result,
                    'agent_type': 'sensor_placement',
                    'csv_file': result.get('csv_info', {}).get('filepath')
                }
            else:
                self.log_error(f"Sensor placement execution failed: {result.get('response', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result.get('response', 'Sensor placement failed'),
                    'step_name': 'Sensor placement'
                }

        except Exception as e:
            error_msg = f"Sensor placement execution error: {str(e)}"
            self.log_error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'step_name': 'Sensor placement'
            }
    
    def _execute_leak_training(self, agent, step_info, conversation_id, user_message):
        """Execute leak model training"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': 'Missing INP file',
                'step_name': 'Leak model training'
            }
        
        # Extract training parameters
        num_scenarios = 50  # Default value
        epochs = 100  # Default value

        # Intelligently extract training parameters
        import re
        num_scenarios, epochs = self._extract_training_parameters(user_message, num_scenarios, epochs)

        self.log_info(f"Parsed training parameters: Sample count={num_scenarios}, Iteration count={epochs}")

        # Call leak detection agent for training
        result = agent.train_leak_detection_model(inp_file, conversation_id, num_scenarios, epochs)
        
        if result['success']:
            self.log_info("Leak model training execution success")
            return {
                'success': True,
                'step_name': 'Leak model training',
                'result': result,
                'agent_type': 'leak_detection',
                'model_file': result.get('files', {}).get('model', {}).get('filepath')
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Leak model training failed'),
                'step_name': 'Leak model training'
            }
    
    def _execute_leak_detection(self, agent, step_info, conversation_id, user_message):
        """Execute leak detection"""
        # Find sensor data file and trained model
        sensor_data_file = self._find_sensor_data_csv(conversation_id)
        model_file = self._find_trained_model(conversation_id)
        
        if not sensor_data_file:
            return {
                'success': False,
                'error': 'Missing sensor data CSV file',
                'step_name': 'Leak detection'
            }
        
        if not model_file:
            return {
                'success': False,
                'error': 'Missing trained leak detection model',
                'step_name': 'Leak detection'
            }
        
        # Call leak detection agent for detection
        result = agent.detect_leak_from_file(sensor_data_file, model_file)
        
        if result['success']:
            self.log_info("Leak detection execution success")
            return {
                'success': True,
                'step_name': 'Leak detection',
                'result': result,
                'agent_type': 'leak_detection'
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Leak detection failed'),
                'step_name': 'Leak detection'
            }
    
    def _execute_generic_step(self, agent, step_info, conversation_id, user_message):
        """Execute generic step"""
        inp_file = step_info.get('input_file')
        if not inp_file:
            return {
                'success': False,
                'error': 'Missing input file',
                'step_name': step_info.get('step_name', 'Unknown step')
            }
        
        # Call agent
        result = agent.process(inp_file, user_message, conversation_id)
        
        return {
            'success': result.get('success', False),
            'step_name': step_info.get('step_name', 'Unknown step'),
            'result': result,
            'error': result.get('response') if not result.get('success') else None
        }
    
    def _find_partition_csv(self, conversation_id: str):
        """Find partition CSV file"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'partition_results' in filename and
                    filename.endswith('.csv')):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_trained_model(self, conversation_id: str):
        """Find trained model"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'leak_detection_model' in filename and
                    filename.endswith('.pth')):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_sensor_data_csv(self, conversation_id: str):
        """Find sensor data CSV file"""
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if (conversation_id[:8] in filename and 
                    filename.endswith('.csv')):
                    return os.path.join(uploads_dir, filename)
        return None
    
    def generate_response_with_llm(self, execution_results: list, original_message: str):
        """Use LLM to generate comprehensive response"""
        try:
            # Build execution result summary
            results_summary = ""
            for i, result in enumerate(execution_results):
                step_name = result.get('step_name', f'Step{i+1}')
                success = result.get('success', False)
                status = "Success" if success else "Failed"
                error = result.get('error', '')
                
                results_summary += f"\n{i+1}. {step_name}: {status}"
                if not success and error:
                    results_summary += f" (Error: {error})"
                elif success and result.get('result'):
                    # Add brief info for successful results
                    agent_result = result['result']
                    if agent_result.get('network_info'):
                        results_summary += f" - Network info analyzed"
                    if agent_result.get('partition_info'):
                        results_summary += f" - Partition results generated"
                    if agent_result.get('sensor_info'):
                        results_summary += f" - Sensor placement completed"
                    if agent_result.get('model_info'):
                        results_summary += f" - Model training completed"
            
            # Build LLM prompt
            prompt = f"""
User original request: "{original_message}"

Execution result summary:
{results_summary}

Please generate a professional, detailed response based on execution results, including:
1. Understanding and confirmation of user request
2. Explanation of execution process
3. Main results and findings
4. If there are failed steps, provide resolution suggestions
5. Next step recommendations (if applicable)

Please respond in professional but understandable language, highlighting key information.

Please use the following signature format at the end of your reply:

Best regards,

Tianwei Mu
Guangzhou Institute of Industrial Intelligence
"""
            
            # Call LLM to generate response
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional water distribution network analysis expert, able to generate clear, professional analysis reports based on agent execution results."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.log_error(f"LLM response generation failed: {e}")
            # Return simple execution summary
            successful_steps = [r['step_name'] for r in execution_results if r.get('success')]
            failed_steps = [r['step_name'] for r in execution_results if not r.get('success')]
            
            summary = f"Execution complete. Successful steps: {', '.join(successful_steps) if successful_steps else 'None'}"
            if failed_steps:
                summary += f". Failed steps: {', '.join(failed_steps)}"
            
            return summary
    
    def process(self, execution_plan: dict, conversation_id: str, user_message: str):
        """Execute complete execution plan"""
        try:
            steps = execution_plan.get('steps', [])
            execution_results = []
            
            self.log_info(f"Starting execution plan, total {len(steps)} steps")
            
            for i, step in enumerate(steps):
                step_name = step.get('step_name')
                step_type = step.get('step_type', 'main')
                
                # Skip prerequisite step (these should be handled at UI layer)
                if step_type == 'prerequisite':
                    self.log_info(f"Skipping prerequisite step: {step_name}")
                    continue
                
                # Execute main step
                result = self.execute_step(step_name, step, conversation_id, user_message)
                execution_results.append(result)
                
                # If step fails and is critical step, may need to stop execution
                if not result.get('success') and step_name in ['Network analysis', 'Network partition']:
                    self.log_warning(f"Critical step failed, stopping execution: {step_name}")
                    break
            
            # Generate comprehensive response
            llm_response = self.generate_response_with_llm(execution_results, user_message)
            
            return {
                'success': True,
                'execution_results': execution_results,
                'llm_response': llm_response,
                'total_steps': len(steps),
                'completed_steps': len(execution_results),
                'execution_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Execution plan failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': datetime.now().isoformat()
            }

    def _extract_training_parameters(self, user_message: str, default_scenarios: int, default_epochs: int) -> tuple:
        """Intelligently extract training parameters"""
        import re

        num_scenarios = default_scenarios
        epochs = default_epochs

        # Pattern for extracting iteration count/training epochs
        epoch_patterns = [
            r'epochs?\s*[=: ]\s*(\d+)',
            r'(\d+)\s*?epochs?',
            r'iteration\s*count\s*[: ]\s*(\d+)',
            r'epoch\s*[=:]\s*(\d+)'
        ]

        # Pattern for extracting sample count/data groups
        scenario_patterns = [
            r'(\d+)Sample',
            r'Sample count\s*[=: ]\s*(\d+)',
            r'Sample\s*(\d+)',
            r'(\d+)\s*samples?',
            r'total\s*samples?\s*[=: ]\s*(\d+)',
            r'sample\s*count\s*[=: ]\s*(\d+)',
            r'sample\s*size\s*[=: ]\s*(\d+)',
            r'generated\s*data\s*[=: ]\s*(\d+)',
            r'number\s*of\s*samples?\s*[=: ]\s*(\d+)'
        ]

        # Try to match iteration count
        for pattern in epoch_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                epochs = min(int(match.group(1)), 500)
                self.log_info(f"Identified iteration count: {epochs} (Matched pattern: {pattern})")
                break

        # Try to match sample count
        for pattern in scenario_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                num_scenarios = min(int(match.group(1)), 2000)  # Increase max limit to 2000
                self.log_info(f"Identified sample count: {num_scenarios} (Matched pattern: {pattern})")
                break

        # If no specific pattern matched, use simple number extraction as fallback
        if num_scenarios == default_scenarios and epochs == default_epochs:
            numbers = re.findall(r'\d+', user_message)
            if numbers:
                # If only one number, determine based on context
                if len(numbers) == 1:
                    num = int(numbers[0])
                    if any(keyword in user_message.lower() for keyword in ['iteration', 'epoch', 'round']):
                        epochs = min(num, 500)
                        self.log_info(f"Identified as iteration count based on context: {epochs}")
                    elif any(keyword in user_message.lower() for keyword in ['packet', 'sample', 'scenario', 'count', 'size']):
                        num_scenarios = min(num, 2000)  # Increase max limit to 2000
                        self.log_info(f"Identified as sample count based on context: {num_scenarios}")
                    else:
                        # Default first number as sample count
                        num_scenarios = min(num, 2000)  # Increase max limit to 2000
                        self.log_info(f"Default identified as sample count: {num_scenarios}")
                elif len(numbers) >= 2:
                    # Multiple numbers: first as sample count, second as iteration count per original logic
                    num_scenarios = min(int(numbers[0]), 2000)  # Increase max limit to 2000
                    epochs = min(int(numbers[1]), 500)
                    self.log_info(f"Multi-number mode: Sample count={num_scenarios}, Iteration count={epochs}")

        return num_scenarios, epochs
