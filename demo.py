from tabular_ai_agent import EnhancedTabularAIAgent 

# Sample demo to run the multi-table pipeline
api_key = ""
agent = EnhancedTabularAIAgent(openrouter_api_key=api_key, model='minimax/minimax-m2:free', use_llm=True) 

html_content = '''
<div class='div-table'>
<!-- First table: Product Sales -->
<table class='pdf-table' border='1' width='100%'>
<tr align="center"><td>Product ID</td><td>Product Name</td><td>Category</td></tr>
<tr align="center"><td>P001</td><td>Laptop</td><td>Electronics</td></tr>
<tr align="center"><td>P002</td><td>Phone</td><td>Electronics</td></tr>
<tr align="center"><td>P003</td><td>Tablet</td><td>Electronics</td></tr>
<tr align="center"><td>P004</td><td>Headphones</td><td>Electronics</td></tr>
</table>
<!-- Second table: Sales by Quarter -->
<table class='pdf-table' border='1' width='100%'>
<tr align="center"><td>Product ID</td><td>Q1 Units</td><td>Q2 Units</td><td>Q3 Units</td></tr>
<tr align="center"><td>P001</td><td>100</td><td>120</td><td>150</td></tr>
<tr align="center"><td>P002</td><td>200</td><td>180</td><td>220</td></tr>
<tr align="center"><td>P003</td><td>80</td><td>90</td><td>110</td></tr>
<tr align="center"><td>P004</td><td>300</td><td>350</td><td>400</td></tr>
</table>
<!-- Third table: Unit Prices -->
<table class='pdf-table' border='1' width='100%'>
<tr align="center"><td>Product ID</td><td>Unit Price</td></tr>
<tr align="center"><td>P001</td><td>1200</td></tr>
<tr align="center"><td>P002</td><td>800</td></tr>
<tr align="center"><td>P003</td><td>600</td></tr>
<tr align="center"><td>P004</td><td>150</td></tr>
</table>
</div>
'''

# Query that requires joining multiple tables
query = 'Calculate the total revenue for each product across all quarters by joining the sales and price tables. Then determine which product had the highest revenue growth rate from Q1 to Q3.'

result = agent.process_query(query, html_content)

_ans = result.final_answer
try:
    print('Answer:', _ans)
except Exception:
    # Fallback for Windows console encoding issues
    safe = _ans.replace('−', '-').replace('×', 'x')
    print('Answer:', safe)
print('Confidence:', result.confidence_score)
print('Processing Time:', result.processing_time)
print('Iterations:', result.iterations)

# Additional info from enhanced version
if result.detected_relationships:
    print(f'\nDetected Relationships: {len(result.detected_relationships)}')
    for rel in result.detected_relationships[:3]:
        print(f"  {rel['table1']}.{rel['column1']} -> {rel['table2']}.{rel['column2']} (confidence: {rel['confidence']:.2f})")

if result.query_plan:
    print(f"\nQuery Plan Complexity: {result.query_plan.get('estimated_complexity', 'N/A')}")
    print(f"Joins Required: {len(result.query_plan.get('required_joins', []))}")

print(f"\nGenerated Code Preview:\n{result.generated_code[:400]}...")
