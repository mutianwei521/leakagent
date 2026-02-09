"""
LLM Task Analyzer
Responsible for converting natural language to agent standard phrases and analyzing task steps
"""
import os
import json
import openai
from datetime import datetime
from .base_agent import BaseAgent

class LLMTaskAnalyzer(BaseAgent):
    """LLM-based task analyzer"""
    
    def __init__(self):
        super().__init__("LLMTaskAnalyzer")
        
        # Set OpenAI API configuration
        openai.api_base = ""
        openai.api_key = ""
        
        # Agent standard phrase mapping
        self.standard_phrases = {
            "Network analysis": "Analyze network structure and basic information",
            "Network partition": "Divide network into specified number of regions",
            "Outlier detection": "Detect and remove outliers from network partition",
            "Sensor placement": "Optimize pressure monitoring sensor placement in network",
            "Resilience analysis": "Analyze resilience and fault detection capability of sensor placement",
            "Leak model training": "Train machine learning-based leak detection model",
            "Leak detection": "Use trained model to detect network leakage",
            "Model inference": "Use trained model for direct inference analysis on sensor data",
            "Hydraulic simulation": "Perform network hydraulic calculation and simulation analysis",
            "Topology analysis": "Analyze network topology structure and connectivity"
        }
        
        # Workflow step definitions
        self.workflow_steps = {
            "complete_workflow": [
                "Network analysis",
                "Network partition", 
                "Outlier detection",
                "Sensor placement",
                "Leak model training"
            ],
            "sensor_placement_workflow": [
                "Network analysis",
                "Network partition",
                "Sensor placement"
            ],
            "leak_detection_workflow": [
                "Network analysis",
                "Network partition",
                "Sensor placement", 
                "Leak model training",
                "Leak detection"
            ]
        }
    
    def analyze_user_intent(self, user_message: str, conversation_history: list = None):
        """Analyze user intent and convert to standard phrase"""
        try:
            # Build analysis prompt
            analysis_prompt = self._build_analysis_prompt(user_message, conversation_history)
            
            # Call LLM for analysis
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional water distribution network agent task analysis expert. You need to convert user's natural language into standard agent phrases and analyze task steps."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content
            return self._parse_llm_response(llm_response, user_message)
            
        except Exception as e:
            self.log_error(f"LLM task analysis failed: {e}")
            return self._fallback_analysis(user_message)
    
    def _build_analysis_prompt(self, user_message: str, conversation_history: list = None):
        """Build analysis prompt"""
        
        # Standard phrase list
        phrases_text = "\n".join([f"- {key}: {value}" for key, value in self.standard_phrases.items()])
        
        # Workflow description
        workflow_text = ""
        for workflow_name, steps in self.workflow_steps.items():
            workflow_text += f"\n{workflow_name}: {' -> '.join(steps)}"
        
        # Conversation history summary
        history_text = ""
        if conversation_history:
            recent_messages = conversation_history[-5:]  # Last 5 messages
            history_text = "\nConversation history summary:\n"
            for i, msg in enumerate(recent_messages):
                history_text += f"{i+1}. User: {msg.get('user', '')[:100]}...\n"
                history_text += f"   Assistant: {msg.get('assistant', '')[:100]}...\n"
        
        prompt = f"""
Please analyze the user's natural language request and complete the following tasks:

1. Convert user request to corresponding agent standard phrase
2. Analyze required task steps
3. Determine if prerequisites need to be checked

Available agent standard phrases:
{phrases_text}

Predefined workflows:
{workflow_text}

User request: "{user_message}"
{history_text}

Please return analysis result in JSON format:
{{
    "standard_phrase": "Corresponding standard phrase",
    "task_type": "Task type(single/workflow)",
    "required_steps": ["Step1", "Step2", ...],
    "prerequisites": {{
        "inp_file": true/false,
        "partition_csv": true/false,
        "trained_model": true/false
    }},
    "parameters": {{
        "num_partitions": number or null,
        "num_sensors": number or null,
        "other_params": "Other parameters"
    }},
    "confidence": 0.0-1.0,
    "explanation": "Analysis description"
}}
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str, original_message: str):
        """Parse LLM response"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['standard_phrase', 'task_type', 'required_steps', 'prerequisites']
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                
                # Add original message
                result['original_message'] = original_message
                result['analysis_time'] = datetime.now().isoformat()
                
                self.log_info(f"LLM analysis successful: {result['standard_phrase']}")
                return result
            else:
                raise ValueError("Cannot extract JSON from LLM response")
                
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            return self._fallback_analysis(original_message)
    
    def _fallback_analysis(self, user_message: str):
        """Fallback analysis method"""
        # Simple keyword matching
        message_lower = user_message.lower()

        if any(word in message_lower for word in ['partition', 'cluster', 'divide']):
            standard_phrase = "Network partition"
            steps = ["Network analysis", "Network partition"]
        elif any(word in message_lower for word in ['sensor', 'monitor', 'place']):
            standard_phrase = "Sensor placement"
            steps = ["Network analysis", "Network partition", "Sensor placement"]
        elif any(word in message_lower for word in ['inference', 'predict', 'analy', 'detect']):
            standard_phrase = "Model inference"
            steps = ["Model inference"]
        elif any(word in message_lower for word in ['leak', 'leakage']):
            if any(word in message_lower for word in ['train', 'model']):
                standard_phrase = "Leak model training"
                steps = ["Network analysis", "Leak model training"]
            else:
                standard_phrase = "Leak detection"
                steps = ["Network analysis", "Network partition", "Sensor placement", "Leak model training", "Leak detection"]
        else:
            standard_phrase = "Network analysis"
            steps = ["Network analysis"]
        
        return {
            "standard_phrase": standard_phrase,
            "task_type": "single" if len(steps) == 1 else "workflow",
            "required_steps": steps,
            "prerequisites": {
                "inp_file": standard_phrase != "Model inference",
                "partition_csv": "Sensor placement" in steps or "Leak detection" in steps,
                "trained_model": "Leak detection" in steps or standard_phrase == "Model inference"
            },
            "parameters": {},
            "confidence": 0.6,
            "explanation": "Fallback analysis using keyword matching",
            "original_message": user_message,
            "analysis_time": datetime.now().isoformat()
        }
    
    def check_prerequisites(self, analysis_result: dict, conversation_id: str):
        """Check task prerequisites"""
        prerequisites = analysis_result.get('prerequisites', {})
        missing_prerequisites = []
        available_files = {}

        try:
            # Check INP file
            if prerequisites.get('inp_file'):
                inp_file = self._find_inp_file(conversation_id)
                if inp_file:
                    available_files['inp_file'] = inp_file
                else:
                    missing_prerequisites.append('inp_file')

            # Intelligent check for partition CSV file - for sensor placement tasks, always check if partition file exists
            standard_phrase = analysis_result.get('standard_phrase', '')
            required_steps = analysis_result.get('required_steps', [])

            # If it's sensor placement related task, proactively check partition file
            if (standard_phrase == 'Sensor placement' or
                'Sensor placement' in required_steps or
                prerequisites.get('partition_csv')):

                partition_csv = self._find_partition_csv(conversation_id)
                if partition_csv:
                    available_files['partition_csv'] = partition_csv
                    self.log_info(f"Detected existing partition file: {partition_csv}")
                else:
                    # Only mark as missing if partition file is explicitly required
                    if prerequisites.get('partition_csv'):
                        missing_prerequisites.append('partition_csv')
                    self.log_info("No partition file detected, will execute complete workflow")

            # Check trained model
            if prerequisites.get('trained_model'):
                model_file = self._find_trained_model(conversation_id)
                if model_file:
                    available_files['trained_model'] = model_file
                else:
                    missing_prerequisites.append('trained_model')

            return {
                'all_satisfied': len(missing_prerequisites) == 0,
                'missing_prerequisites': missing_prerequisites,
                'available_files': available_files,
                'check_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error(f"Check prerequisites failed: {e}")
            return {
                'all_satisfied': False,
                'missing_prerequisites': list(prerequisites.keys()),
                'available_files': {},
                'error': str(e),
                'check_time': datetime.now().isoformat()
            }
    
    def _find_inp_file(self, conversation_id: str):
        """Find conversation-related INP file"""
        # Search in uploads directory
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if (conversation_id[:8] in filename and 
                    filename.endswith('.inp') and 
                    os.path.exists(os.path.join(uploads_dir, filename))):
                    return os.path.join(uploads_dir, filename)
        return None
    
    def _find_partition_csv(self, conversation_id: str):
        """Find conversation-related partition CSV file"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'partition_results' in filename and
                    filename.endswith('.csv') and
                    os.path.exists(os.path.join(downloads_dir, filename))):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def _find_trained_model(self, conversation_id: str):
        """Find conversation-related trained model"""
        downloads_dir = 'downloads'
        if os.path.exists(downloads_dir):
            for filename in os.listdir(downloads_dir):
                if (conversation_id[:8] in filename and 
                    'leak_detection_model' in filename and
                    filename.endswith('.pth') and
                    os.path.exists(os.path.join(downloads_dir, filename))):
                    return os.path.join(downloads_dir, filename)
        return None
    
    def generate_execution_plan(self, analysis_result: dict, prerequisites_check: dict):
        """Generate execution plan"""
        required_steps = analysis_result.get('required_steps', [])
        missing_prerequisites = prerequisites_check.get('missing_prerequisites', [])
        available_files = prerequisites_check.get('available_files', {})
        standard_phrase = analysis_result.get('standard_phrase', '')

        execution_plan = {
        # Total steps will be updated at the end
            'steps': [],
            'estimated_time': 0,
            'plan_time': datetime.now().isoformat(),
            'workflow_type': 'incremental' if 'partition_csv' in available_files else 'complete'
        }

        # Intelligent flow control: special handling for sensor placement tasks
        if standard_phrase == 'Sensor placement':
            if 'partition_csv' in available_files:
                # Case 1: Already has partition file, directly perform sensor placement
                self.log_info("Detected partition file, will skip partition step and directly perform sensor placement")
                execution_plan['workflow_description'] = "Perform sensor placement based on existing partition results"

                # Only add sensor placement step
                execution_plan['steps'].append({
                    'step_name': 'Sensor placement',
                    'step_type': 'main',
                    'description': 'Optimize pressure monitoring sensor placement based on existing partition results',
                    'estimated_minutes': 5,
                    'order': 1,
                    'input_file': available_files.get('inp_file'),
                    'partition_file': available_files['partition_csv']
                })

            else:
                # Case 2: No partition file, execute complete workflow
                self.log_info("No partition file detected, will execute complete analysis->partition->sensor placement workflow")
                execution_plan['workflow_description'] = "Execute complete network analysis, partition and sensor placement workflow"

                # Add complete workflow steps
                if 'inp_file' in available_files:
                    execution_plan['steps'].extend([
                        {
                            'step_name': 'Network analysis',
                            'step_type': 'main',
                            'description': 'Analyze network structure and basic information',
                            'estimated_minutes': 2,
                            'order': 1,
                            'input_file': available_files['inp_file']
                        },
                        {
                            'step_name': 'Network partition',
                            'step_type': 'main',
                            'description': 'Divide network into specified number of regions',
                            'estimated_minutes': 3,
                            'order': 2,
                            'input_file': available_files['inp_file']
                        },
                        {
                            'step_name': 'Sensor placement',
                            'step_type': 'main',
                            'description': 'Optimize pressure monitoring sensor placement in network',
                            'estimated_minutes': 5,
                            'order': 3,
                            'input_file': available_files['inp_file']
                        }
                    ])
                else:
                    # Missing INP file
                    execution_plan['steps'].append({
                        'step_name': 'Upload INP file',
                        'step_type': 'prerequisite',
                        'description': 'Need to upload network INP file to continue',
                        'estimated_minutes': 1
                    })

            # Update total steps and calculate estimated time
            execution_plan['total_steps'] = len(execution_plan['steps'])
            execution_plan['estimated_time'] = sum(step.get('estimated_minutes', 0) for step in execution_plan['steps'])

            return execution_plan

        # Original logic for other tasks
        # If there are missing prerequisites, need to execute prerequisite steps first
        if missing_prerequisites:
            if 'inp_file' in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': 'Upload INP file',
                    'step_type': 'prerequisite',
                    'description': 'Need to upload network INP file to continue',
                    'estimated_minutes': 1
                })

            if 'partition_csv' in missing_prerequisites and 'inp_file' not in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': 'Network partition',
                    'step_type': 'prerequisite',
                    'description': 'Need to perform network partition first to generate CSV file',
                    'estimated_minutes': 3
                })

            if 'trained_model' in missing_prerequisites:
                execution_plan['steps'].append({
                    'step_name': 'Train leak model',
                    'step_type': 'prerequisite',
                    'description': 'Need to train leak detection model first',
                    'estimated_minutes': 10
                })
        
        # Add main execution steps, skip already completed steps
        for i, step in enumerate(required_steps):
            # If partition file already exists, skip network partition step
            if step == 'Network partition' and 'partition_csv' in available_files:
                self.log_info(f"Skipping step '{step}' - partition file exists: {available_files['partition_csv']}")
                continue

            # If trained model already exists, skip model training step
            if step == 'Leak model training' and 'trained_model' in available_files:
                self.log_info(f"Skipping step '{step}' - trained model exists: {available_files['trained_model']}")
                continue

            step_info = {
                'step_name': step,
                'step_type': 'main',
                'description': self.standard_phrases.get(step, step),
                'estimated_minutes': self._estimate_step_time(step),
                'order': i + 1
            }
            
            # Add available files info
            if step == 'Network analysis' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Network partition' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Hydraulic simulation' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Topology analysis' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Leak model training' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Sensor placement' and 'inp_file' in available_files:
                step_info['input_file'] = available_files['inp_file']
            elif step == 'Leak detection' and 'trained_model' in available_files:
                step_info['input_file'] = available_files['trained_model']
            
            execution_plan['steps'].append(step_info)
        
        # Update total steps and calculate estimated time
        execution_plan['total_steps'] = len(execution_plan['steps'])
        execution_plan['estimated_time'] = sum(step.get('estimated_minutes', 0) for step in execution_plan['steps'])

        return execution_plan
    
    def _estimate_step_time(self, step_name: str):
        """Estimate step execution time (minutes)"""
        time_estimates = {
            "Network analysis": 2,
            "Network partition": 3,
            "Outlier detection": 2,
            "Sensor placement": 5,
            "Resilience analysis": 3,
            "Leak model training": 10,
            "Leak detection": 2,
            "Hydraulic simulation": 3,
            "Topology analysis": 2
        }
        return time_estimates.get(step_name, 3)
    
    def process(self, user_message: str, conversation_id: str, conversation_history: list = None):
        """Main method to process user message"""
        try:
            # 1. Analyze user intent
            analysis_result = self.analyze_user_intent(user_message, conversation_history)
            
            # 2. Check prerequisites
            prerequisites_check = self.check_prerequisites(analysis_result, conversation_id)
            
            # 3. Generate execution plan
            execution_plan = self.generate_execution_plan(analysis_result, prerequisites_check)
            
            # 4. Combine results
            result = {
                'success': True,
                'analysis': analysis_result,
                'prerequisites': prerequisites_check,
                'execution_plan': execution_plan,
                'conversation_id': conversation_id,
                'process_time': datetime.now().isoformat()
            }
            
            self.log_info(f"Task analysis complete: {analysis_result['standard_phrase']}")
            return result
            
        except Exception as e:
            self.log_error(f"Failed to process task: {e}")
            return {
                'success': False,
                'error': str(e),
                'conversation_id': conversation_id,
                'process_time': datetime.now().isoformat()
            }
