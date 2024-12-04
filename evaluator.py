# evaluator.py
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import logging

class ChatbotEvaluator:
    def __init__(self, chatbot, test_cases_path: str = 'test.json'):
        """Initialize evaluator with chatbot instance and test cases"""
        self.chatbot = chatbot
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Then load test cases
        self.test_cases = self._load_test_cases(test_cases_path)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'details': []
        }

    def _load_test_cases(self, path: str) -> List[Dict]:
        """Load test cases from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            self.logger.info(f"Loaded {len(test_cases)} test cases")
            return test_cases
        except FileNotFoundError:
            self.logger.warning(f"Test cases file not found: {path}")
            return []
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON in test cases file: {path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading test cases: {e}")
            return []

    def evaluate_response_quality(self, response: str, expected: str, query: str) -> Dict:
        """Evaluate response quality using LLM"""
        eval_prompt = f"""You are evaluating a university chatbot response. 
Return ONLY a JSON object in exactly this format (no extra text or explanations):

{{
    "factual_accuracy": 0.0,
    "completeness": 0.0,
    "relevance": 0.0,
    "clarity": 0.0,
    "hallucination": 0.0
}}

Use decimal numbers between 0.0 and 1.0 for each score, note i should have like 0.76, 0.2, 0.9.

Query: {query}
Expected Response: {expected}
Actual Response: {response}"""

        try:
            # Get model response
            model_evaluation = self.chatbot.get_response(eval_prompt)
            print(model_evaluation)
            
            # Parse JSON, stripping any extra text
            response_text = model_evaluation['response'].strip()
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response_text[first_brace:last_brace + 1]
                scores = json.loads(json_str)
                
                # Convert all values to float
                return {k: float(v) for k, v in scores.items()}
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            self.logger.error(f"Error in response evaluation: {e}")
            return {
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'relevance': 0.0,
                'clarity': 0.0,
                'hallucination': 0.0
            }

    def evaluate_retrieval(self, response_data: Dict, test_case: Dict) -> Dict:
        """Evaluate knowledge retrieval using LLM"""
        retrieval_prompt = f"""Evaluate the knowledge retrieval performance.
Return ONLY a JSON object in exactly this format (no extra text or explanations):

{{
    "precision": 0.0,
    "recall": 0.0,
    "relevance": 0.0
}}

Use decimal numbers between 0.0 and 1.0 for each score.

Retrieved Knowledge:
{json.dumps(response_data.get('relevant_knowledge', []), indent=2)}

Expected Knowledge:
{json.dumps(test_case.get('expected_knowledge', []), indent=2)}

Query: {test_case['query']}"""

        try:
            # Get model response
            evaluation = self.chatbot.get_response(retrieval_prompt)
            
            # Parse JSON, stripping any extra text
            response_text = evaluation['response'].strip()
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response_text[first_brace:last_brace + 1]
                scores = json.loads(json_str)
                
                # Convert all values to float
                return {k: float(v) for k, v in scores.items()}
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            self.logger.error(f"Error in retrieval evaluation: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'relevance': 0.0
            }

    def evaluate_overall_quality(self, response_data: Dict, test_case: Dict) -> Dict:
        """Evaluate overall response quality"""
        quality_prompt = f"""Evaluate this chatbot interaction.
Return ONLY a JSON object in exactly this format (no extra text or explanations):

{{
    "overall_quality": 0.0,
    "confidence_alignment": 0.0,
    "format_quality": 0.0
}}

Use decimal numbers between 0.0 and 1.0 for each score.

Query: {test_case['query']}
Type: {test_case['type']}
Response: {response_data.get('response', '')}
Confidence: {response_data.get('confidence', 0)}"""

        try:
            # Get model response
            evaluation = self.chatbot.get_response(quality_prompt)
            
            # Parse JSON, stripping any extra text
            response_text = evaluation['response'].strip()
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json_str = response_text[first_brace:last_brace + 1]
                scores = json.loads(json_str)
                
                # Convert all values to float
                return {k: float(v) for k, v in scores.items()}
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            self.logger.error(f"Error in quality evaluation: {e}")
            return {
                'overall_quality': 0.0,
                'confidence_alignment': 0.0,
                'format_quality': 0.0
            }

    def run_evaluation(self) -> Dict:
        """Run full evaluation suite"""
        start_time = time.time()
        
        for test_case in self.test_cases:
            try:
                test_start_time = time.time()
                
                # Get chatbot response
                response_data = self.chatbot.get_response(test_case['query'])
                
                # Evaluate different aspects
                response_quality = self.evaluate_response_quality(
                    response_data.get('response', ''),
                    test_case['expected_response'],
                    test_case['query']
                )
                
                retrieval_metrics = self.evaluate_retrieval(response_data, test_case)
                
                overall_quality = self.evaluate_overall_quality(response_data, test_case)
                
                # Calculate performance metrics
                performance_metrics = {
                    'response_time': time.time() - test_start_time,
                    'success': response_data.get('status') == 'success'
                }
                
                # Store detailed results
                self.results['details'].append({
                    'test_case': test_case,
                    'response': response_data,
                    'metrics': {
                        'response_quality': response_quality,
                        'retrieval': retrieval_metrics,
                        'quality': overall_quality,  # Changed from overall_quality to quality
                        'performance': performance_metrics
                    }
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating test case: {e}")
                continue
        
        # Calculate aggregate metrics
        try:
            self._calculate_aggregate_metrics()
        except Exception as e:
            self.logger.error(f"Error calculating aggregate metrics: {e}")
            
        self.results['total_time'] = time.time() - start_time
        return self.results

    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics from all test cases"""
        categories = {
            'response_quality': ['factual_accuracy', 'completeness', 'relevance', 'clarity', 'hallucination'],
            'retrieval': ['precision', 'recall', 'relevance'],
            'quality': ['overall_quality', 'confidence_alignment', 'format_quality']
        }
        
        # Initialize aggregates for metrics
        aggregates = {category: {metric: [] for metric in metrics} 
                    for category, metrics in categories.items()}
        
        # Initialize a list to track scores of each metric type
        all_metric_scores = {
            'factual_accuracy': [],
            'completeness': [],
            'relevance': [],
            'clarity': [],
            'hallucination': []
        }
        
        # Collect metrics
        for detail in self.results['details']:
            metrics = detail.get('metrics', {})
            test_case = detail.get('test_case', {})
            
            # Process response quality metrics
            quality_metrics = metrics.get('response_quality', {})
            for metric in all_metric_scores.keys():
                if metric in quality_metrics:
                    all_metric_scores[metric].append(quality_metrics[metric])
            
            # Process category metrics - only include if expected knowledge exists
            for category, metric_list in categories.items():
                category_metrics = metrics.get(category, {})
                if isinstance(category_metrics, dict):
                    # For retrieval metrics, only include if expected_knowledge exists and isn't empty
                    if category == 'retrieval' and not test_case.get('expected_knowledge', []):
                        continue
                    
                    for metric in metric_list:
                        value = category_metrics.get(metric)
                        if isinstance(value, (int, float)):
                            aggregates[category][metric].append(float(value))
        
        # Calculate averages for individual metrics
        self.results['metrics'] = {
            category: {
                metric: sum(values) / len(values) if values else 0.0 
                for metric, values in metrics.items()
            }
            for category, metrics in aggregates.items()
        }
        
        # Calculate overall averages for each evaluator type
        self.results['evaluator_averages'] = {
            category: sum(
                sum(values) / len(values) if values else 0.0 
                for values in aggregates[category].values()
            ) / len(categories[category])
            for category in categories
        }
        
        # Calculate average scores for each metric type
        self.results['metric_type_averages'] = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in all_metric_scores.items()
        }
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        try:
            # Calculate averages before saving
            if 'evaluator_averages' not in self.results:
                self._calculate_aggregate_metrics()
                
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def generate_report(self) -> str:
        """Generate human-readable evaluation report"""
        if not self.results.get('metrics'):
            return "No evaluation results available"

        report = [
            "Chatbot Evaluation Report",
            f"Generated: {self.results['timestamp']}",
            f"Total Evaluation Time: {self.results.get('total_time', 0):.2f} seconds\n",
            "Overall Metrics:",
            "-" * 50
        ]

        # Add evaluator averages
        if 'evaluator_averages' in self.results:
            report.extend([
                "\nEvaluator Averages:",
                "-" * 30
            ])
            for evaluator, average in self.results['evaluator_averages'].items():
                report.append(f"{evaluator.replace('_', ' ').title()}: {average:.3f}")

        # Add aggregate metrics
        report.append("\nDetailed Metrics by Category:")
        report.append("-" * 30)
        for category, metrics in self.results['metrics'].items():
            report.append(f"\n{category.title()} Metrics:")
            for metric, value in metrics.items():
                report.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

        # Add detailed test cases
        report.extend([
            "\nDetailed Test Cases:",
            "-" * 50
        ])

        for i, detail in enumerate(self.results['details'], 1):
            report.extend([
                f"\nTest Case {i}:",
                f"Query: {detail['test_case']['query']}",
                f"Expected: {detail['test_case']['expected_response']}",
                f"Actual: {detail['response'].get('response', '')}",
                f"Response Time: {detail['metrics']['performance']['response_time']:.3f}s",
                f"Success: {detail['metrics']['performance']['success']}"
            ])

        return "\n".join(report)
    
if __name__ == "__main__":
    from chatbot import UniversityChatbot
    
    # Initialize chatbot and evaluator
    chatbot = UniversityChatbot(model_name='gpt2')
    evaluator = ChatbotEvaluator(chatbot)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Generate and print report
    print(evaluator.generate_report())
    
    # Save results
    evaluator.save_results()