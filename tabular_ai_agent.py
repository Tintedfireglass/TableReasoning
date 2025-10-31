from openai import OpenAI
import pandas as pd
import numpy as np
import re
import json
import ast
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    query: str
    refined_queries: List[str]
    generated_code: str
    execution_result: Any
    final_answer: str
    confidence_score: float
    processing_time: float
    iterations: int

@dataclass
class TableSchema:
    columns: List[str]
    data_types: Dict[str, str]
    sample_data: Dict[str, List[Any]]
    metadata: Dict[str, Any]
    html_structure: Dict[str, Any]

class TabularAIAgent:
    def __init__(self, openrouter_api_key: str, model: str = "qwen/qwen3-coder:free"):
        """
        Initialize the Tabular AI Agent using OpenRouter API
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.model = model
        self.cache = {}
        self.max_iterations = 5
        self.confidence_threshold = 0.8

        # Optional headers for OpenRouter ranking
        self.extra_headers = {
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional
            "X-Title": "<YOUR_SITE_NAME>",      # Optional
        }

        # Prompts
        self.prompts = {
            'schema_generation': """
            Analyze the following HTML table and generate a comprehensive schema.
            Focus on:
            1. Column names and their semantic meaning
            2. Data types (string, number, date, boolean)
            3. Sample values for each column
            4. Relationships between columns
            5. Any special formatting or patterns

            HTML Table:
            {html_table}

            Return JSON format:
            {{
                "columns": ["col1", "col2", ...],
                "data_types": {{"col1": "string", "col2": "number"}},
                "sample_data": {{"col1": ["val1", "val2"], "col2": [1, 2]}},
                "metadata": {{"description": "...", "patterns": [...]}},
                "html_structure": {{"headers": [...], "footnotes": [...]} }
            }}
            """,
            'query_refinement': """
            Given the table schema and user query, break down the query into specific,
            actionable sub-queries that can be answered with code.

            Table Schema: {schema}
            User Query: {query}

            Generate 2-4 refined sub-queries that:
            1. Are specific and measurable
            2. Can be answered with pandas operations
            3. Cover all aspects of the original query
            4. Are ordered logically

            Return as JSON array of strings.
            """,
            'code_generation': """
            Generate Python pandas code to answer the following sub-queries.
            Use the provided table schema and sample data.

            Schema: {schema}
            Sub-queries: {sub_queries}

            Requirements:
            1. Use pandas DataFrame operations
            2. Handle missing data appropriately
            3. Include data type conversions if needed
            4. Add comments explaining each step
            5. Return results in a clear format
            6. Generate ONLY executable Python code – NO markdown formatting
            7. Do NOT include explanatory text outside of comments
            8. Ensure all strings are properly quoted

            Generate complete, executable Python code:
            """,
            'reflection': """
            Analyze the execution results and determine if the task is complete.

            Original Query: {original_query}
            Generated Code: {code}
            Execution Results: {results}

            Evaluate:
            1. Does the result fully answer the original query?
            2. Are there any errors or unexpected results?
            3. What improvements could be made?
            4. Confidence score (0-1) in the current answer

            Return ONLY valid JSON format:
            {{
                "is_complete": true/false,
                "confidence": 0.0-1.0,
                "issues": ["issue1", "issue2"],
                "improvements": ["improvement1", "improvement2"],
                "next_steps": "description of what to do next"
            }}
            """,
            'answer_synthesis': """
            Synthesize the execution results into a clear, natural language answer.

            Original Query: {query}
            Execution Results: {results}
            Confidence: {confidence}

            Create a comprehensive answer that:
            1. Directly addresses the user's question
            2. Includes relevant data and statistics
            3. Explains the reasoning process
            4. Highlights any limitations or uncertainties

            Format as clear, professional text.
            """
        }

    def extract_tables_from_html(self, html_content: str) -> List[pd.DataFrame]:
        """
        Extract tables from HTML content and convert to pandas DataFrames

        Args:
            html_content: HTML string containing tables

        Returns:
            List of pandas DataFrames
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        dataframes = []
        for i, table in enumerate(tables):
            try:
                rows = table.find_all('tr')
                if not rows:
                    continue

                processed_rows = self._process_table_rows(rows)
                if not processed_rows:
                    continue

                headers = processed_rows[0]
                data_rows = processed_rows[1:] if len(processed_rows) > 1 else []

                if headers and data_rows:
                    max_cols = max(len(headers), max(len(row) for row in data_rows) if data_rows else 0)

                    # Pad headers if needed
                    while len(headers) < max_cols:
                        headers.append(f"Column_{len(headers)}")

                    padded_data_rows = []
                    for row in data_rows:
                        padded_row = row[:]
                        while len(padded_row) < max_cols:
                            padded_row.append("")
                        padded_data_rows.append(padded_row)

                    df = pd.DataFrame(padded_data_rows, columns=headers)
                    dataframes.append(df)
                    logger.info(f"Extracted table {i+1} with shape {df.shape}")

            except Exception as e:
                logger.error(f"Error extracting table {i+1}: {e}")
                continue

        return dataframes

    def _process_table_rows(self, rows) -> List[List[str]]:
        """
        Process table rows to handle rowspan and colspan attributes

        Args:
            rows: List of BeautifulSoup tr elements

        Returns:
            List of processed row data
        """
        processed_rows = []

        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = []

            for cell in cells:
                text = cell.get_text(strip=True)
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))

                row_data.append(text)
                for _ in range(colspan - 1):
                    row_data.append("")

            if row_data:
                processed_rows.append(row_data)

        return processed_rows

    def generate_table_schema(self, df: pd.DataFrame, html_table: str) -> TableSchema:
        """
        Generate comprehensive schema for a DataFrame

        Args:
            df: pandas DataFrame
            html_table: Original HTML table string

        Returns:
            TableSchema object
        """
        try:
            data_types = {}
            sample_data = {}

            # Handle duplicate column names
            df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                          for i, col in enumerate(df.columns)]

            for col in df.columns:
                try:
                    col_series = df[col]
                    if isinstance(col_series, pd.Series):
                        dtype = col_series.dtype
                        if dtype in ['int64', 'float64']:
                            data_types[col] = 'number'
                        elif dtype == 'bool':
                            data_types[col] = 'boolean'
                        elif pd.api.types.is_datetime64_any_dtype(col_series):
                            data_types[col] = 'datetime'
                        else:
                            data_types[col] = 'string'
                    else:
                        data_types[col] = 'string'

                    try:
                        if isinstance(col_series, pd.Series):
                            sample_data[col] = col_series.dropna().head(5).tolist()
                        else:
                            sample_data[col] = []
                    except Exception as sample_error:
                        logger.warning(f"Error getting sample data for column '{col}': {sample_error}")
                        sample_data[col] = []

                except Exception as col_error:
                    logger.warning(f"Error processing column {col}: {col_error}")
                    data_types[col] = 'string'
                    sample_data[col] = []

            soup = BeautifulSoup(html_table, 'html.parser')
            html_structure = {
                'headers': [th.get_text(strip=True) for th in soup.find_all('th')],
                'footnotes': [sup.get_text(strip=True) for sup in soup.find_all('sup')],
                'rowspan_colspan': self._analyze_table_structure(soup)
            }

            metadata = {
                'shape': df.shape,
                'description': f"Table with {df.shape[0]} rows and {df.shape[1]} columns",
                'patterns': self._identify_patterns(df),
                'extraction_time': datetime.now().isoformat()
            }

            return TableSchema(
                columns=list(df.columns),
                data_types=data_types,
                sample_data=sample_data,
                metadata=metadata,
                html_structure=html_structure
            )

        except Exception as e:
            logger.error(f"Error generating schema: {e}")
            raise

    def _analyze_table_structure(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Analyze HTML table structure for rowspan/colspan"""
        structure_info = {
            'has_rowspan': False,
            'has_colspan': False,
            'complex_cells': []
        }

        for cell in soup.find_all(['td', 'th']):
            if cell.get('rowspan'):
                structure_info['has_rowspan'] = True
            if cell.get('colspan'):
                structure_info['has_colspan'] = True

            if cell.get('rowspan') or cell.get('colspan'):
                structure_info['complex_cells'].append({
                    'text': cell.get_text(strip=True),
                    'rowspan': cell.get('rowspan'),
                    'colspan': cell.get('colspan')
                })

        return structure_info

    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify patterns in the data"""
        patterns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            patterns.append(f"Numeric columns: {list(numeric_cols)}")

        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10)
                if any(re.match(r'\d{4}-\d{2}-\d{2}', str(val)) for val in sample_values):
                    patterns.append(f"Date-like pattern in column: {col}")

        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].dtype == 'object':
                patterns.append(f"Categorical column: {col}")

        return patterns

    def refine_query(self, query: str, schema: TableSchema) -> List[str]:
        """
        Break down user query into specific sub-queries
        """
        try:
            prompt = self.prompts['query_refinement'].format(
                schema=json.dumps(schema.__dict__, indent=2),
                query=query
            )
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            refined_queries = json.loads(response.choices[0].message.content)
            logger.info(f"Refined queries: {refined_queries}")
            return refined_queries
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return [query]

    def generate_code(self, sub_queries: List[str], schema: TableSchema) -> str:
        """
        Generate Python pandas code to answer the sub-queries
        """
        try:
            prompt = self.prompts['code_generation'].format(
                schema=json.dumps(schema.__dict__, indent=2),
                sub_queries=json.dumps(sub_queries, indent=2)
            )
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            code = response.choices[0].message.content

            # Clean up markdown/code fences
            lines = code.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.strip() in ['```python', '```', '```py']:
                    continue
                if (line.strip().startswith('This code') or line.strip().startswith('The code')
                    or line.strip().startswith('Please note') or line.strip().startswith('This code first')
                    or line.strip().startswith('Finally') or line.strip().startswith('The DataFrame')
                    or line.strip().startswith('It then') or line.strip().startswith('The columns')
                    or line.strip().startswith('The unnamed')):
                    continue
                cleaned_lines.append(line)
            code = '\n'.join(cleaned_lines).strip()
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            code = code.strip()

            logger.info(f"Generated code: {code[:200]}...")
            return code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def execute_code(self, code: str, df: pd.DataFrame) -> Tuple[Any, str]:
        """
        Execute generated Python code in a safe sandbox environment
        """
        try:
            safe_globals = {
                'pd': pd,
                'np': np,
                'df': df,
                'DataFrame': pd.DataFrame,
                'Series': pd.Series,
                'json': json,
                're': re,
                'datetime': datetime,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'sum': sum,
                'max': max,
                'min': min,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'range': range,
                'print': print
            }
            safe_locals = {}

            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                exec(code, safe_globals, safe_locals)

            result = None
            for var_name in ['result', 'answer', 'output', 'data']:
                if var_name in safe_locals:
                    result = safe_locals[var_name]
                    break

            if result is None:
                try:
                    lines = code.strip().split('\n')
                    last_line = lines[-1].strip()
                    if last_line and not last_line.startswith('#'):
                        result = eval(last_line, safe_globals, safe_locals)
                except Exception:
                    result = "Code executed successfully"

            output = output_buffer.getvalue()
            comprehensive_result = {
                "result": result,
                "output": output,
                "variables": {k: str(v)[:200] for k, v in safe_locals.items() if not k.startswith('_')}
            }
            logger.info(f"Code execution completed. Output: {output[:100]}...")
            return comprehensive_result, output

        except Exception as e:
            error_msg = f"Code execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg

    def reflect_on_results(self, original_query: str, code: str, results: Any, output: str) -> Dict[str, Any]:
        """
        Analyze the execution results and determine if the task is complete
        """
        try:
            prompt = self.prompts['reflection'].format(
                original_query=original_query,
                code=code,
                results=str(results)[:500]
            )
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            reflection_text = response.choices[0].message.content.strip()
            if reflection_text.startswith('```json'):
                reflection_text = reflection_text[7:]
            if reflection_text.endswith('```'):
                reflection_text = reflection_text[:-3]
            reflection_text = reflection_text.strip()
            reflection = json.loads(reflection_text)
            logger.info(f"Reflection: {reflection}")
            return reflection
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return {
                "is_complete": True,
                "confidence": 0.5,
                "issues": [f"Reflection error: {e}"],
                "improvements": [],
                "next_steps": "Continue with current results"
            }

    def synthesize_answer(self, query: str, results: Any, confidence: float) -> str:
        """
        Synthesize the execution results into a clear, natural language answer
        """
        try:
            prompt = self.prompts['answer_synthesis'].format(
                query=query,
                results=str(results)[:1000],
                confidence=confidence
            )
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            logger.info(f"Synthesized answer: {answer[:200]}...")
            return answer
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return f"Based on the analysis: {str(results)}"

    def process_query(self, query: str, html_content: str) -> QueryResult:
        """
        Entry point to process a user query on HTML table content
        """
        start_time = datetime.now()
        try:
            dataframes = self.extract_tables_from_html(html_content)
            if not dataframes:
                raise ValueError("No tables found in HTML content")

            df = dataframes[0]
            if df.empty:
                raise ValueError("Extracted table is empty")

            schema = self.generate_table_schema(df, html_content)
            refined_queries = self.refine_query(query, schema)

            best_result = None
            best_confidence = 0.0
            iterations = 0

            for iteration in range(self.max_iterations):
                iterations += 1
                logger.info(f"Starting iteration {iteration+1}")
                code = self.generate_code(refined_queries, schema)
                results, output = self.execute_code(code, df)
                reflection = self.reflect_on_results(query, code, results, output)

                if reflection.get('confidence', 0) > best_confidence:
                    best_result = {
                        'code': code,
                        'results': results,
                        'output': output,
                        'reflection': reflection
                    }
                    best_confidence = reflection.get('confidence', 0)

                if reflection.get('is_complete', False) and best_confidence >= self.confidence_threshold:
                    break

                if not reflection.get('is_complete', False) and iteration < self.max_iterations - 1:
                    improvements = reflection.get('improvements', [])
                    if improvements:
                        refined_queries = improvements

            final_answer = self.synthesize_answer(query, best_result['results'], best_confidence)
            processing_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                query=query,
                refined_queries=refined_queries,
                generated_code=best_result['code'],
                execution_result=best_result['results'],
                final_answer=final_answer,
                confidence_score=best_confidence,
                processing_time=processing_time,
                iterations=iterations
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=query,
                refined_queries=[],
                generated_code="",
                execution_result=None,
                final_answer=f"Error processing query: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                iterations=0
            )
