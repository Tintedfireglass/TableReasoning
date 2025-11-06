from openai import OpenAI
import pandas as pd
import numpy as np
import re
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import io
from contextlib import redirect_stdout, redirect_stderr
from difflib import SequenceMatcher

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
    query_plan: Optional[Dict[str, Any]] = None
    detected_relationships: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class TableSchema:
    columns: List[str]
    data_types: Dict[str, str]
    sample_data: Dict[str, List[Any]]
    metadata: Dict[str, Any]
    html_structure: Dict[str, Any]
    statistical_profile: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relationship:
    table1: str
    table2: str
    column1: str
    column2: str
    relationship_type: str
    confidence: float
    match_method: str
    sample_overlap: int = 0
    overlap_percentage: float = 0.0

@dataclass
class MultiTableSchema:
    tables: List[Dict[str, Any]]
    relationships: List[Relationship]
    total_tables: int
    relationship_graph: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class QueryPlan:
    steps: List[Dict[str, Any]]
    estimated_complexity: str
    required_joins: List[Dict[str, Any]]
    aggregations: List[str]
    filters: List[str]
    reasoning: str

class EnhancedTabularAIAgent:
    def __init__(self, openrouter_api_key: str, model: str = "qwen/qwen3-coder:free"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        self.model = model
        self.cache = {}
        self.max_iterations = 5
        self.confidence_threshold = 0.8
        self.max_tables = 10
        self.similarity_threshold = 0.7
        self.value_overlap_threshold = 0.3
        
        self.extra_headers = {
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>",
        }

        self.prompts = {
            'semantic_column_matching': """
            Analyze these two columns from different tables and determine if they are semantically related.
            
            Table 1: {table1_name}, Column: {column1_name}
            Type: {column1_type}, Samples: {column1_samples}
            
            Table 2: {table2_name}, Column: {column2_name}
            Type: {column2_type}, Samples: {column2_samples}
            
            Return ONLY valid JSON:
            {{
                "are_related": true/false,
                "confidence": 0.0-1.0,
                "relationship_type": "exact_match|synonym|foreign_key|unrelated",
                "reasoning": "Brief explanation",
                "recommended_join": true/false,
                "join_type": "inner|left|right|outer|none"
            }}
            """,
            
            'multi_column_relationship_analysis': """
            Analyze relationships between multiple columns across tables.
            
            Tables Info: {tables_info}
            Programmatic Matches: {programmatic_matches}
            
            Return ONLY valid JSON:
            {{
                "validated_relationships": [
                    {{
                        "table1": "...", "table2": "...",
                        "columns": {{"table1": ["col1"], "table2": ["col2"]}},
                        "is_valid": true, "confidence": 0.9,
                        "relationship_type": "foreign_key", "reasoning": "..."
                    }}
                ],
                "new_relationships": [],
                "composite_keys": [],
                "ranked_relationships": []
            }}
            """,
            
            'join_strategy_recommendation': """
            Recommend the best join strategy for this query.
            
            User Query: {query}
            Tables: {tables_summary}
            Relationships: {relationships}
            
            Return ONLY valid JSON:
            {{
                "recommended_joins": [
                    {{
                        "step": 1, "left_table": "table1", "right_table": "table2",
                        "join_type": "inner", "join_keys": {{"left": ["col1"], "right": ["col2"]}},
                        "reasoning": "..."
                    }}
                ],
                "pre_join_filters": [],
                "potential_issues": [],
                "alternative_approaches": []
            }}
            """,
            
            'query_planning': """
            Create a detailed execution plan for this query.
            
            Schema: {schema}
            Query: {query}
            Relationships: {relationships}
            
            Return ONLY valid JSON:
            {{
                "steps": [
                    {{"step": 1, "action": "load_table", "table": "table1", "reason": "..."}}
                ],
                "estimated_complexity": "simple|moderate|complex",
                "required_joins": [],
                "aggregations": [],
                "filters": [],
                "reasoning": "Overall plan explanation"
            }}
            """,
            
            'query_refinement': """
            Break down the user query into specific sub-queries.
            
            Schema: {schema}
            Query: {query}
            
            Return JSON array of 2-4 specific, measurable sub-queries.
            """,
            
            'reflection': """
            Analyze execution results and determine completeness.
            
            Query: {original_query}
            Code: {code}
            Results: {results}
            
            Return ONLY valid JSON:
            {{
                "is_complete": true/false,
                "confidence": 0.0-1.0,
                "issues": ["..."],
                "improvements": ["..."],
                "next_steps": "..."
            }}
            """,
            
            'answer_synthesis': """
            Synthesize results into a clear answer.
            
            Query: {query}
            Results: {results}
            Confidence: {confidence}
            
            Create a comprehensive, professional answer with data and reasoning.
            """
        }

    def extract_tables_from_html(self, html_content: str) -> List[pd.DataFrame]:
        """Extract tables from HTML"""
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
                    
                    while len(headers) < max_cols:
                        headers.append(f"Column_{len(headers)}")
                    
                    padded_data_rows = []
                    for row in data_rows:
                        padded_row = row[:]
                        while len(padded_row) < max_cols:
                            padded_row.append("")
                        padded_data_rows.append(padded_row)
                    
                    df = pd.DataFrame(padded_data_rows, columns=headers)
                    df.columns = [self._clean_column_name(col) for col in df.columns]
                    dataframes.append(df)
                    logger.info(f"Extracted table {i+1} with shape {df.shape}")
            
            except Exception as e:
                logger.error(f"Error extracting table {i+1}: {e}")
                continue
        
        return dataframes

    def _clean_column_name(self, col: str) -> str:
        """Clean column names"""
        col = str(col).strip()
        col = re.sub(r'\s+', '_', col)
        col = re.sub(r'[^\w\s-]', '', col)
        return col

    def _process_table_rows(self, rows) -> List[List[str]]:
        """Process table rows"""
        processed_rows = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            row_data = []
            for cell in cells:
                text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                row_data.append(text)
                for _ in range(colspan - 1):
                    row_data.append("")
            if row_data:
                processed_rows.append(row_data)
        return processed_rows

    def generate_enhanced_table_schema(self, df: pd.DataFrame, html_table: str, 
                                      table_name: str) -> TableSchema:
        """Generate enhanced schema with statistics"""
        df.columns = [f"{col}_{i}" if df.columns.tolist().count(col) > 1 else col 
                      for i, col in enumerate(df.columns)]
        
        data_types = {}
        sample_data = {}
        statistical_profile = {}
        
        for col in df.columns:
            try:
                col_series = df[col]
                
                # Data type
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
                
                # Sample data
                if isinstance(col_series, pd.Series):
                    sample_data[col] = col_series.dropna().head(5).tolist()
                else:
                    sample_data[col] = []
                
                # Statistics
                if isinstance(col_series, pd.Series):
                    profile = {
                        'null_count': int(col_series.isna().sum()),
                        'null_percentage': float(col_series.isna().sum() / len(col_series) * 100),
                        'unique_count': int(col_series.nunique()),
                        'unique_percentage': float(col_series.nunique() / len(col_series) * 100)
                    }
                    
                    if data_types[col] == 'number':
                        numeric_col = pd.to_numeric(col_series, errors='coerce')
                        if not numeric_col.isna().all():
                            profile.update({
                                'min': float(numeric_col.min()),
                                'max': float(numeric_col.max()),
                                'mean': float(numeric_col.mean()),
                                'median': float(numeric_col.median())
                            })
                    
                    statistical_profile[col] = profile
            
            except Exception as e:
                logger.warning(f"Error processing column {col}: {e}")
                data_types[col] = 'string'
                sample_data[col] = []
                statistical_profile[col] = {}
        
        soup = BeautifulSoup(html_table, 'html.parser')
        html_structure = {
            'headers': [th.get_text(strip=True) for th in soup.find_all('th')],
            'footnotes': [sup.get_text(strip=True) for sup in soup.find_all('sup')]
        }
        
        metadata = {
            'table_name': table_name,
            'shape': df.shape,
            'description': f"Table with {df.shape[0]} rows and {df.shape[1]} columns",
            'extraction_time': datetime.now().isoformat()
        }
        
        return TableSchema(
            columns=list(df.columns),
            data_types=data_types,
            sample_data=sample_data,
            metadata=metadata,
            html_structure=html_structure,
            statistical_profile=statistical_profile
        )

    def _calculate_value_overlap(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """Calculate overlap between two series"""
        try:
            values1 = set(series1.dropna().unique())
            values2 = set(series2.dropna().unique())
            
            if not values1 or not values2:
                return {'common_count': 0, 'overlap_percentage': 0.0}
            
            common = values1.intersection(values2)
            overlap_pct = (len(common) / min(len(values1), len(values2))) * 100
            
            return {'common_count': len(common), 'overlap_percentage': overlap_pct}
        except Exception:
            return {'common_count': 0, 'overlap_percentage': 0.0}

    def _determine_relationship_type(self, series1: pd.Series, series2: pd.Series) -> str:
        """Determine relationship type"""
        try:
            unique1 = series1.nunique()
            unique2 = series2.nunique()
            total1 = len(series1)
            total2 = len(series2)
            
            ratio1 = unique1 / total1 if total1 > 0 else 0
            ratio2 = unique2 / total2 if total2 > 0 else 0
            
            if ratio1 > 0.9 and ratio2 > 0.9:
                return 'one_to_one'
            elif ratio1 > 0.9 or ratio2 > 0.9:
                return 'one_to_many'
            else:
                return 'many_to_many'
        except Exception:
            return 'unknown'

    def _find_exact_column_matches(self, schema1: Dict, schema2: Dict, 
                                   df1: pd.DataFrame, df2: pd.DataFrame) -> List[Relationship]:
        """Find exact column name matches"""
        relationships = []
        cols1 = set(schema1['columns'])
        cols2 = set(schema2['columns'])
        common_cols = cols1.intersection(cols2)
        
        for col in common_cols:
            is_key = any(kw in col.lower() for kw in ['id', 'code', 'key', 'number', 'name'])
            
            if is_key or schema1['data_types'].get(col) in ['number', 'string']:
                rel_type = self._determine_relationship_type(df1[col], df2[col])
                overlap = self._calculate_value_overlap(df1[col], df2[col])
                
                relationships.append(Relationship(
                    table1=schema1['metadata']['table_name'],
                    table2=schema2['metadata']['table_name'],
                    column1=col,
                    column2=col,
                    relationship_type=rel_type,
                    confidence=0.9 if is_key else 0.7,
                    match_method='exact_name',
                    sample_overlap=overlap['common_count'],
                    overlap_percentage=overlap['overlap_percentage']
                ))
        
        return relationships

    def _find_semantic_column_matches(self, schema1: Dict, schema2: Dict,
                                     df1: pd.DataFrame, df2: pd.DataFrame) -> List[Relationship]:
        """Find semantic column matches"""
        relationships = []
        
        for col1 in schema1['columns']:
            for col2 in schema2['columns']:
                if col1 == col2:
                    continue
                
                similarity = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
                
                if similarity >= self.similarity_threshold:
                    overlap = self._calculate_value_overlap(df1[col1], df2[col2])
                    
                    if overlap['overlap_percentage'] >= self.value_overlap_threshold * 100:
                        rel_type = self._determine_relationship_type(df1[col1], df2[col2])
                        
                        relationships.append(Relationship(
                            table1=schema1['metadata']['table_name'],
                            table2=schema2['metadata']['table_name'],
                            column1=col1,
                            column2=col2,
                            relationship_type=rel_type,
                            confidence=similarity * (overlap['overlap_percentage'] / 100),
                            match_method='semantic',
                            sample_overlap=overlap['common_count'],
                            overlap_percentage=overlap['overlap_percentage']
                        ))
        
        return relationships

    def _find_value_overlap_matches(self, schema1: Dict, schema2: Dict,
                                   df1: pd.DataFrame, df2: pd.DataFrame) -> List[Relationship]:
        """Find value overlap matches"""
        relationships = []
        
        for col1 in schema1['columns']:
            for col2 in schema2['columns']:
                if col1 == col2:
                    continue
                
                type1 = schema1['data_types'].get(col1)
                type2 = schema2['data_types'].get(col2)
                
                if type1 != type2:
                    continue
                
                overlap = self._calculate_value_overlap(df1[col1], df2[col2])
                
                if overlap['overlap_percentage'] >= self.value_overlap_threshold * 100:
                    rel_type = self._determine_relationship_type(df1[col1], df2[col2])
                    
                    relationships.append(Relationship(
                        table1=schema1['metadata']['table_name'],
                        table2=schema2['metadata']['table_name'],
                        column1=col1,
                        column2=col2,
                        relationship_type=rel_type,
                        confidence=overlap['overlap_percentage'] / 100,
                        match_method='value_overlap',
                        sample_overlap=overlap['common_count'],
                        overlap_percentage=overlap['overlap_percentage']
                    ))
        
        return relationships

    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships"""
        seen = {}
        for rel in relationships:
            key = tuple(sorted([f"{rel.table1}.{rel.column1}", f"{rel.table2}.{rel.column2}"]))
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        return sorted(list(seen.values()), key=lambda x: x.confidence, reverse=True)

    def detect_relationships_advanced(self, dataframes: List[pd.DataFrame], 
                                     table_schemas: List[Dict[str, Any]]) -> List[Relationship]:
        """Advanced relationship detection"""
        relationships = []
        
        for i, (df1, schema1) in enumerate(zip(dataframes, table_schemas)):
            for j, (df2, schema2) in enumerate(zip(dataframes[i+1:], table_schemas[i+1:]), start=i+1):
                exact_matches = self._find_exact_column_matches(schema1, schema2, df1, df2)
                relationships.extend(exact_matches)
                
                semantic_matches = self._find_semantic_column_matches(schema1, schema2, df1, df2)
                relationships.extend(semantic_matches)
                
                value_matches = self._find_value_overlap_matches(schema1, schema2, df1, df2)
                relationships.extend(value_matches)
        
        relationships = self._deduplicate_relationships(relationships)
        return relationships

    def llm_semantic_column_match(self, table1_info: Dict, col1: str, 
                                  table2_info: Dict, col2: str) -> Dict[str, Any]:
        """Use LLM for semantic column matching"""
        try:
            prompt = self.prompts['semantic_column_matching'].format(
                table1_name=table1_info['metadata']['table_name'],
                column1_name=col1,
                column1_type=table1_info['data_types'].get(col1, 'unknown'),
                column1_samples=json.dumps(table1_info['sample_data'].get(col1, [])[:5]),
                table2_name=table2_info['metadata']['table_name'],
                column2_name=col2,
                column2_type=table2_info['data_types'].get(col2, 'unknown'),
                column2_samples=json.dumps(table2_info['sample_data'].get(col2, [])[:5])
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            logger.info(f"LLM match {col1}-{col2}: {result['are_related']} ({result['confidence']})")
            return result
        
        except Exception as e:
            logger.error(f"Error in LLM semantic matching: {e}")
            return {
                "are_related": False,
                "confidence": 0.0,
                "relationship_type": "error",
                "reasoning": str(e),
                "recommended_join": False,
                "join_type": "none"
            }

    def llm_validate_relationships(self, table_schemas: List[Dict[str, Any]], 
                                   programmatic_matches: List[Relationship]) -> Dict[str, Any]:
        """Use LLM to validate relationships"""
        try:
            tables_info = []
            for schema in table_schemas:
                table_summary = {
                    'name': schema['metadata']['table_name'],
                    'columns': schema['columns'][:10],
                    'data_types': schema['data_types'],
                    'sample_data': {k: v[:3] for k, v in list(schema['sample_data'].items())[:5]}
                }
                tables_info.append(table_summary)
            
            matches_summary = [
                {
                    'table1': r.table1,
                    'table2': r.table2,
                    'column1': r.column1,
                    'column2': r.column2,
                    'match_method': r.match_method,
                    'confidence': r.confidence
                }
                for r in programmatic_matches[:10]
            ]
            
            prompt = self.prompts['multi_column_relationship_analysis'].format(
                tables_info=json.dumps(tables_info, indent=2),
                programmatic_matches=json.dumps(matches_summary, indent=2)
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            logger.info(f"LLM validated {len(result.get('validated_relationships', []))} relationships")
            return result
        
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return {
                "validated_relationships": [],
                "new_relationships": [],
                "composite_keys": [],
                "ranked_relationships": []
            }

    def llm_recommend_join_strategy(self, query: str, table_schemas: List[Dict[str, Any]],
                                    relationships: List[Relationship]) -> Dict[str, Any]:
        """Use LLM to recommend join strategy"""
        try:
            tables_summary = [
                {
                    'name': schema['metadata']['table_name'],
                    'columns': schema['columns'],
                    'row_count': schema['metadata']['shape'][0]
                }
                for schema in table_schemas
            ]
            
            relationships_summary = [
                {
                    'table1': r.table1,
                    'table2': r.table2,
                    'column1': r.column1,
                    'column2': r.column2,
                    'type': r.relationship_type,
                    'confidence': r.confidence
                }
                for r in relationships[:10]
            ]
            
            prompt = self.prompts['join_strategy_recommendation'].format(
                query=query,
                tables_summary=json.dumps(tables_summary, indent=2),
                relationships=json.dumps(relationships_summary, indent=2)
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            logger.info(f"LLM recommended {len(result.get('recommended_joins', []))} joins")
            return result
        
        except Exception as e:
            logger.error(f"Error in join strategy: {e}")
            return {
                "recommended_joins": [],
                "pre_join_filters": [],
                "potential_issues": [],
                "alternative_approaches": []
            }

    def detect_relationships_hybrid(self, dataframes: List[pd.DataFrame], 
                                    table_schemas: List[Dict[str, Any]],
                                    use_llm_validation: bool = True,
                                    use_llm_discovery: bool = True) -> List[Relationship]:
        """Hybrid relationship detection"""
        logger.info("Running programmatic detection...")
        programmatic_relationships = self.detect_relationships_advanced(dataframes, table_schemas)
        logger.info(f"Found {len(programmatic_relationships)} programmatic relationships")
        
        if not use_llm_validation and not use_llm_discovery:
            return programmatic_relationships
        
        logger.info("Using LLM to validate and enhance...")
        llm_analysis = self.llm_validate_relationships(table_schemas, programmatic_relationships)
        
        final_relationships = []
        
        # Add validated relationships
        validated = llm_analysis.get('validated_relationships', [])
        for v in validated:
            if v.get('is_valid', False):
                orig = next((r for r in programmatic_relationships 
                           if r.table1 == v['table1'] and r.table2 == v['table2']), None)
                
                if orig:
                    boosted_confidence = (orig.confidence + v['confidence']) / 2
                    final_relationships.append(Relationship(
                        table1=orig.table1,
                        table2=orig.table2,
                        column1=orig.column1,
                        column2=orig.column2,
                        relationship_type=v.get('relationship_type', orig.relationship_type),
                        confidence=boosted_confidence,
                        match_method=f"{orig.match_method}+llm_validated",
                        sample_overlap=orig.sample_overlap,
                        overlap_percentage=orig.overlap_percentage
                    ))
        
        # Add LLM-discovered relationships
        if use_llm_discovery:
            new_rels = llm_analysis.get('new_relationships', [])
            for n in new_rels:
                if n.get('confidence', 0) >= 0.6:
                    col1 = n['columns']['table1'][0] if n['columns']['table1'] else None
                    col2 = n['columns']['table2'][0] if n['columns']['table2'] else None
                    
                    if col1 and col2:
                        final_relationships.append(Relationship(
                            table1=n['table1'],
                            table2=n['table2'],
                            column1=col1,
                            column2=col2,
                            relationship_type=n.get('relationship_type', 'llm_discovered'),
                            confidence=n['confidence'],
                            match_method='llm_discovered',
                            sample_overlap=0,
                            overlap_percentage=0.0
                        ))
        
        if not final_relationships and programmatic_relationships:
            logger.warning("No validated relationships, keeping top programmatic")
            final_relationships = programmatic_relationships[:5]
        
        logger.info(f"Final relationship count: {len(final_relationships)}")
        return final_relationships

    def create_query_plan(self, query: str, schema: MultiTableSchema, 
                         relationships: List[Relationship]) -> QueryPlan:
        """Create query execution plan"""
        
        prompt = self.prompts['query_planning'].format(
            schema=json.dumps(schema.__dict__, default=str, indent=2)[:2000],
            query=query,
            relationships=json.dumps([r.__dict__ for r in relationships], indent=2)[:1000]
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            extra_headers=self.extra_headers,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        plan_text = response.choices[0].message.content.strip()
        plan_text = plan_text.replace('```json', '').replace('```', '').strip()
        plan_dict = json.loads(plan_text)
        
        query_plan = QueryPlan(
            steps=plan_dict.get('steps', []),
            estimated_complexity=plan_dict.get('estimated_complexity', 'moderate'),
            required_joins=plan_dict.get('required_joins', []),
            aggregations=[],
            filters=[],
            reasoning="Using fallback plan"
        )
        return query_plan


    def generate_advanced_code(self, query_plan: QueryPlan, schema: MultiTableSchema,
                              relationships: List[Relationship], query: str,
                              num_dataframes: int) -> str:
        """Generate advanced code based on query plan"""
        try:
            available_dfs = [f"df{i+1}" for i in range(num_dataframes)]
            
            code_prompt = f"""
            Generate Python pandas code based on this plan:
            
            Query: {query}
            Available DataFrames: {', '.join(available_dfs)}
            Query Plan: {json.dumps(query_plan.__dict__, indent=2)[:1000]}
            Relationships: {json.dumps([r.__dict__ for r in relationships], indent=2)[:500]}
            
            Requirements:
            1. Follow the query plan steps
            2. Use efficient pandas operations
            3. Handle missing data
            4. Store result in 'result' variable
            5. Add comments
            6. Generate ONLY executable Python code - NO markdown
            
            Generate complete Python code:
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.2
            )
            
            code = response.choices[0].message.content
            code = self._clean_generated_code(code)
            
            logger.info(f"Generated code: {code[:200]}...")
            return code
        
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def _clean_generated_code(self, code: str) -> str:
        """Clean generated code"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped in ['```python', '```', '```py']:
                continue
            if any(stripped.startswith(phrase) for phrase in [
                'This code', 'The code', 'Please note', 'Finally',
                'The DataFrame', 'It then', 'The columns'
            ]):
                continue
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        code = code.replace('```python', '').replace('```', '').strip()
        return code

    def execute_code_safely(self, code: str, dataframes: List[pd.DataFrame]) -> Tuple[Any, str]:
        """Execute code safely"""
        try:
            safe_globals = {
                'pd': pd,
                'np': np,
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
                'print': print,
                'abs': abs,
                'round': round
            }
            
            # Add dataframes
            for i, df in enumerate(dataframes[:self.max_tables]):
                safe_globals[f'df{i+1}'] = df
            
            if len(dataframes) == 1:
                safe_globals['df'] = dataframes[0]
            
            safe_locals = {}
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                exec(code, safe_globals, safe_locals)
            
            # Find result
            result = None
            for var_name in ['result', 'answer', 'output', 'data', 'final_result']:
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
                "variables": {k: str(v)[:200] for k, v in safe_locals.items() 
                            if not k.startswith('_')}
            }
            
            logger.info("Code executed successfully")
            return comprehensive_result, output
        
        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg

    def reflect_on_results(self, original_query: str, query_plan: Optional[QueryPlan], 
                          code: str, results: Any, output: str) -> Dict[str, Any]:
        """Reflect on execution results"""
        try:
            prompt = self.prompts['reflection'].format(
                original_query=original_query,
                code=code[:500],
                results=str(results)[:500]
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.extra_headers,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            reflection_text = response.choices[0].message.content.strip()
            reflection_text = reflection_text.replace('```json', '').replace('```', '').strip()
            reflection = json.loads(reflection_text)
            
            logger.info(f"Reflection - Confidence: {reflection.get('confidence', 0)}")
            return reflection
        
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            return {
                "is_complete": True,
                "confidence": 0.5,
                "issues": [f"Reflection error: {e}"],
                "improvements": [],
                "next_steps": "Continue"
            }

    def synthesize_answer(self, query: str, results: Any, confidence: float, 
                         query_plan: Optional[QueryPlan] = None) -> str:
        """Synthesize final answer"""
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
            logger.info("Answer synthesized")
            return answer
        
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Based on analysis: {str(results)[:500]}"

    def _execute_with_reflection(self, query: str, refined_queries: List[str],
                                schema_info: Dict[str, Any], dataframes: List[pd.DataFrame],
                                query_plan: Optional[QueryPlan] = None,
                                relationships: Optional[List[Relationship]] = None,
                                start_time: datetime = None) -> Dict[str, Any]:
        """Unified execution pipeline with reflection"""
        if start_time is None:
            start_time = datetime.now()
        
        is_multi_table = len(dataframes) > 1
        best_result = None
        best_confidence = 0.0
        iterations = 0
        
        logger.info(f"Unified pipeline ({'multi' if is_multi_table else 'single'}-table)")
        
        for iteration in range(self.max_iterations):
            iterations += 1
            logger.info(f"Iteration {iteration+1}/{self.max_iterations}")
            
            # Generate code
            try:
                if is_multi_table:
                    code = self.generate_advanced_code(
                        query_plan, schema_info, relationships, query, len(dataframes)
                    )
                else:
                    code_prompt = f"""
                    Generate Python pandas code using DataFrame 'df':
                    
                    Query: {query}
                    Sub-queries: {json.dumps(refined_queries)}
                    
                    Requirements:
                    1. Efficient pandas operations
                    2. Handle missing data
                    3. Store result in 'result' variable
                    4. Generate ONLY executable code - NO markdown
                    
                    Generate code:
                    """
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        extra_headers=self.extra_headers,
                        messages=[{"role": "user", "content": code_prompt}],
                        temperature=0.2
                    )
                    code = self._clean_generated_code(response.choices[0].message.content)
                
                logger.info(f"Code generated (iteration {iteration+1})")
            
            except Exception as e:
                logger.error(f"Code generation error: {e}")
                continue
            
            # Execute code
            try:
                results, output = self.execute_code_safely(code, dataframes)
                logger.info(f"Code executed (iteration {iteration+1})")
            except Exception as e:
                logger.error(f"Execution error: {e}")
                results = None
                output = str(e)
            
            # Reflect
            try:
                reflection = self.reflect_on_results(query, query_plan, code, results, output)
                current_confidence = reflection.get('confidence', 0)
                logger.info(f"Confidence: {current_confidence}")
            except Exception as e:
                logger.error(f"Reflection error: {e}")
                reflection = {
                    "is_complete": False,
                    "confidence": 0.3,
                    "issues": [str(e)],
                    "improvements": [],
                    "next_steps": "Try again"
                }
                current_confidence = 0.3
            
            # Track best
            if current_confidence > best_confidence:
                best_result = {
                    'code': code,
                    'results': results,
                    'output': output,
                    'reflection': reflection
                }
                best_confidence = current_confidence
                logger.info(f"New best confidence: {best_confidence}")
            
            # Check completion
            if reflection.get('is_complete', False) and best_confidence >= self.confidence_threshold:
                logger.info(f"Completed in {iterations} iterations")
                break
            
            # Refine
            if not reflection.get('is_complete', False) and iteration < self.max_iterations - 1:
                improvements = reflection.get('improvements', [])
                if improvements:
                    logger.info(f"Refining: {improvements}")
                    refined_queries = improvements
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'best_result': best_result,
            'best_confidence': best_confidence,
            'iterations': iterations,
            'processing_time': processing_time,
            'refined_queries': refined_queries
        }

    def _process_single_table(self, query: str, html_content: str, 
                             df: pd.DataFrame, start_time: datetime) -> QueryResult:
        """Process single table query"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            html_table = str(soup.find('table'))
            
            schema = self.generate_enhanced_table_schema(df, html_table, 'table1')
            
            prompt = self.prompts['query_refinement'].format(
                schema=json.dumps(schema.__dict__, default=str, indent=2)[:2000],
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
            
            execution_result = self._execute_with_reflection(
                query=query,
                refined_queries=refined_queries,
                schema_info=schema.__dict__,
                dataframes=[df],
                query_plan=None,
                relationships=None,
                start_time=start_time
            )
            
            best_result = execution_result['best_result']
            best_confidence = execution_result['best_confidence']
            iterations = execution_result['iterations']
            processing_time = execution_result['processing_time']
            
            final_answer = self.synthesize_answer(query, best_result['results'], best_confidence)
            
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
            logger.error(f"Single table error: {e}")
            logger.error(traceback.format_exc())
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=query,
                refined_queries=[],
                generated_code="",
                execution_result=None,
                final_answer=f"Error: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                iterations=0
            )

    def _process_multi_table_enhanced(self, query: str, html_content: str,
                                     dataframes: List[pd.DataFrame], 
                                     start_time: datetime) -> QueryResult:
        """Process multi-table query"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            html_tables = soup.find_all('table')
            
            table_schemas = []
            for i, (df, html_table) in enumerate(zip(dataframes, html_tables)):
                schema = self.generate_enhanced_table_schema(df, str(html_table), f'table{i+1}')
                table_info = {
                    'name': f'table{i+1}',
                    'index': i,
                    **schema.__dict__
                }
                table_schemas.append(table_info)
            
            logger.info("Hybrid relationship detection...")
            relationships = self.detect_relationships_hybrid(
                dataframes, table_schemas,
                use_llm_validation=True,
                use_llm_discovery=True
            )
            logger.info(f"Detected {len(relationships)} relationships")
            
            multi_schema = MultiTableSchema(
                tables=table_schemas,
                relationships=relationships,
                total_tables=len(dataframes)
            )
            
            logger.info("Getting join strategy...")
            join_strategy = self.llm_recommend_join_strategy(query, table_schemas, relationships)
            
            logger.info("Creating query plan...")
            query_plan = self.create_query_plan(query, multi_schema, relationships)
            
            refined_queries = [step.get('action', '') for step in query_plan.steps]
            
            execution_result = self._execute_with_reflection(
                query=query,
                refined_queries=refined_queries,
                schema_info=multi_schema,
                dataframes=dataframes,
                query_plan=query_plan,
                relationships=relationships,
                start_time=start_time
            )
            
            best_result = execution_result['best_result']
            best_confidence = execution_result['best_confidence']
            iterations = execution_result['iterations']
            processing_time = execution_result['processing_time']
            
            final_answer = self.synthesize_answer(
                query, best_result['results'], best_confidence, query_plan
            )
            
            result = QueryResult(
                query=query,
                refined_queries=refined_queries,
                generated_code=best_result['code'],
                execution_result=best_result['results'],
                final_answer=final_answer,
                confidence_score=best_confidence,
                processing_time=processing_time,
                iterations=iterations,
                query_plan=query_plan.__dict__,
                detected_relationships=[r.__dict__ for r in relationships]
            )
            
            if hasattr(result, '__dict__'):
                result.__dict__['join_strategy'] = join_strategy
            
            return result
        
        except Exception as e:
            logger.error(f"Multi-table error: {e}")
            logger.error(traceback.format_exc())
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=query,
                refined_queries=[],
                generated_code="",
                execution_result=None,
                final_answer=f"Error: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                iterations=0
            )

    def process_query(self, query: str, html_content: str) -> QueryResult:
        """Main entry point for query processing"""
        start_time = datetime.now()
        
        try:
            dataframes = self.extract_tables_from_html(html_content)
            
            if not dataframes:
                raise ValueError("No tables found in HTML")
            
            if any(df.empty for df in dataframes):
                raise ValueError("One or more tables are empty")
            
            logger.info(f"Extracted {len(dataframes)} tables")
            
            if len(dataframes) > 1:
                logger.info("Using multi-table processing")
                return self._process_multi_table_enhanced(query, html_content, dataframes, start_time)
            else:
                logger.info("Using single-table processing")
                return self._process_single_table(query, html_content, dataframes[0], start_time)
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=query,
                refined_queries=[],
                generated_code="",
                execution_result=None,
                final_answer=f"Error: {str(e)}",
                confidence_score=0.0,
                processing_time=processing_time,
                iterations=0
            )


