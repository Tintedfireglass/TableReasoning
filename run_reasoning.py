import sys
from pathlib import Path
from tabular_ai_agent import EnhancedTabularAIAgent
if len(sys.argv) < 4:
    print("Usage: python idk.py <html_file> <api_key> <model_name>")
    sys.exit(1)
html_path = Path(sys.argv[1])
api_key = sys.argv[2]
model_name = sys.argv[3]
if not html_path.exists():
    print(f"HTML file not found: {html_path}")
    sys.exit(1)
with open(html_path, "r") as f:
    html_content = f.read()
# print(f"model: {model_name}")
agent = EnhancedTabularAIAgent(api_key, model=model_name)
query = 'Calculate the total revenue for each product across all quarters by joining the sales and price tables. Then determine which product had the highest revenue growth rate from Q1 to Q3.'
result = agent.process_query(query, html_content)
print('Answer:', result.final_answer)
print('Confidence:', result.confidence_score)
print('Processing Time:', result.processing_time)
print('Iterations:', result.iterations)
if result.detected_relationships:
    print(f'\nDetected Relationships: {len(result.detected_relationships)}')
    for rel in result.detected_relationships[:3]:
        print(f"  {rel['table1']}.{rel['column1']} ↔ {rel['table2']}.{rel['column2']} (confidence: {rel['confidence']:.2f})")

if result.query_plan:
    print(f"\nQuery Plan Complexity: {result.query_plan.get('estimated_complexity', 'N/A')}")
    print(f"Joins Required: {len(result.query_plan.get('required_joins', []))}")

print(f"\nGenerated Code Preview:\n{result.generated_code[:400]}...")
