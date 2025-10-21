from tabular_ai_agent import TabularAIAgent

# Sample demo to run the pipeline
api_key = "ENTER_API_KEY_HERE"
agent = TabularAIAgent(api_key, model='gpt-4')

html_content = '''
<div class='div-table'>
<table class='pdf-table' border='1' width='100%'>
<tr align=\"center\"><td width=\"20%\">Product</td><td width=\"20%\">Q1 Units</td><td width=\"20%\">Q2 Units</td><td width=\"20%\">Q3 Units</td><td width=\"20%\">Unit Price</td></tr>
<tr align=\"center\"><td width=\"20%\">Laptop</td><td width=\"20%\">100</td><td width=\"20%\">120</td><td width=\"20%\">150</td><td width=\"20%\">1200</td></tr>
<tr align=\"center\"><td width=\"20%\">Phone</td><td width=\"20%\">200</td><td width=\"20%\">180</td><td width=\"20%\">220</td><td width=\"20%\">800</td></tr>
<tr align=\"center\"><td width=\"20%\">Tablet</td><td width=\"20%\">80</td><td width=\"20%\">90</td><td width=\"20%\">110</td><td width=\"20%\">600</td></tr>
<tr align=\"center\"><td width=\"20%\">Headphones</td><td width=\"20%\">300</td><td width=\"20%\">350</td><td width=\"20%\">400</td><td width=\"20%\">150</td></tr>
</table>
</div>
'''

result = agent.process_query('Calculate the total revenue for each product across all quarters and determine which product had the highest revenue growth rate from Q1 to Q3', html_content)
print('Answer:', result.final_answer)
print('Confidence:', result.confidence_score)
print('Processing Time:', result.processing_time)
print('Iterations:', result.iterations)
