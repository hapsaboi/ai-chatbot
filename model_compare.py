import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import logging
from datetime import datetime
from chatbot import UniversityChatbot
from evaluator import ChatbotEvaluator

class ModelComparison:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_to_test = [
            # None,  # Default Mistral API
            # 'mistralai/Pixtral-Large-Instruct-2411',
            # 'mistralai/Mistral-7B-Instruct-v0.1'
            'HuggingFaceH4/zephyr-7b-beta'
        ]
        
    def run_comparison(self):
        """Run evaluations across all models and compare results"""
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'model_results': {},
            'failed_models': []
        }
        
        successful_evaluations = False
        
        for model_name in self.models_to_test:
            model_id = model_name if model_name else 'mistral-default'
            print(f"\nEvaluating model: {model_id}")
            
            try:
                # Initialize chatbot with specific model
                chatbot = UniversityChatbot(model_name=model_name)
                evaluator = ChatbotEvaluator(chatbot)
                
                # Run evaluation
                results = evaluator.run_evaluation()
                
                if results and results.get('evaluator_averages'):
                    comparison_results['model_results'][model_id] = {
                        'response_quality': results['evaluator_averages']['response_quality'],
                        'retrieval': results['evaluator_averages']['retrieval'],
                        'quality': results['evaluator_averages']['quality'],
                        'metric_type_averages': results['metric_type_averages'],
                        'total_time': results['total_time']
                    }
                    successful_evaluations = True
                    
                    # Save individual results
                    evaluator.save_results(f'evaluation_results_{model_id}.json')
                else:
                    raise ValueError("No valid evaluation metrics obtained")
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error evaluating {model_id}: {error_msg}")
                self.logger.error(f"Model evaluation failed for {model_id}: {error_msg}")
                comparison_results['failed_models'].append({
                    'model': model_id,
                    'error': error_msg
                })
                continue
        
        if successful_evaluations:
            try:
                # Generate comparison plots
                if comparison_results['model_results']:
                    self.plot_model_comparison(comparison_results)
                
                # Save comparison results
                with open('model_comparison_results.json', 'w') as f:
                    json.dump(comparison_results, f, indent=2)
                
                return comparison_results
            except Exception as e:
                self.logger.error(f"Error saving comparison results: {e}")
                return comparison_results  # Return results even if saving fails
        else:
            print("No successful model evaluations completed")
            return None
            
    def plot_model_comparison(self, comparison_results):
        """Generate comparison plots for all models"""
        if not comparison_results.get('model_results'):
            self.logger.warning("No results to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 1. Main Metrics Comparison
        metrics = ['response_quality', 'retrieval', 'quality']
        models = list(comparison_results['model_results'].keys())
        
        x = np.arange(len(metrics))
        width = 0.2
        multiplier = 0
        
        ax = plt.subplot(211)
        
        for model_name, results in comparison_results['model_results'].items():
            offset = width * multiplier
            values = [results[metric] for metric in metrics]
            rects = ax.bar(x + offset, values, width, label=model_name)
            
            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom')
            
            multiplier += 1
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right')
        plt.xticks(rotation=45)
        
        # 2. Detailed Metrics Comparison
        ax2 = plt.subplot(212)
        detailed_metrics = ['factual_accuracy', 'completeness', 'relevance', 'clarity', 'hallucination']
        x = np.arange(len(detailed_metrics))
        multiplier = 0
        
        for model_name, results in comparison_results['model_results'].items():
            offset = width * multiplier
            values = [results['metric_type_averages'][metric] for metric in detailed_metrics]
            rects = ax2.bar(x + offset, values, width, label=model_name)
            
            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                ax2.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom')
            
            multiplier += 1
        
        ax2.set_ylabel('Score')
        ax2.set_title('Detailed Metrics Comparison')
        ax2.set_xticks(x + width * (len(models) - 1) / 2)
        ax2.set_xticklabels(detailed_metrics)
        ax2.legend(loc='upper right')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison_plots.png', bbox_inches='tight', dpi=300)
        plt.close()

    def generate_comparison_report(self, results):
        """Generate a detailed comparison report"""
        if not results:
            return "No comparison results available"
            
        report = [
            "Model Comparison Report",
            "=====================",
            f"Generated: {results['timestamp']}\n"
        ]
        
        # Report failed models
        if results.get('failed_models'):
            report.extend([
                "Failed Models:",
                "-------------"
            ])
            for failure in results['failed_models']:
                report.append(f"{failure['model']}: {failure['error']}")
            report.append("")
        
        # Report successful models
        if results['model_results']:
            report.append("Successful Model Evaluations:")
            report.append("---------------------------")
            for model, res in results['model_results'].items():
                report.extend([
                    f"\nModel: {model}",
                    "-" * (len(model) + 7),
                    f"Main Metrics:",
                    f"- Response Quality: {res['response_quality']:.2%}",
                    f"- Retrieval: {res['retrieval']:.2%}",
                    f"- Quality: {res['quality']:.2%}",
                    f"- Total Time: {res['total_time']:.2f}s\n",
                    f"Detailed Metrics:"
                ])
                
                for metric, value in res['metric_type_averages'].items():
                    report.append(f"- {metric}: {value:.2%}")
                
                report.append("")
        else:
            report.append("\nNo successful model evaluations to report.")
        
        return "\n".join(report)

def main():
    try:
        # Initialize and run comparison
        comparison = ModelComparison()
        results = comparison.run_comparison()
        
        if results:
            # Generate and save report
            report = comparison.generate_comparison_report(results)
            try:
                with open('model_comparison_report.txt', 'w') as f:
                    f.write(report)
                
                print("\nComparison complete! Generated files:")
                if results.get('model_results'):
                    print("- model_comparison_plots.png")
                print("- model_comparison_results.json")
                print("- model_comparison_report.txt")
            except Exception as e:
                print(f"\nError saving report files: {e}")
                
            print("\nReport Preview:")
            print("================")
            print(report)
        else:
            print("\nNo results generated due to evaluation failures")
    except Exception as e:
        print(f"Error running comparison: {e}")

if __name__ == "__main__":
    main()