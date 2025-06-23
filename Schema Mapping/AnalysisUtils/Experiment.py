import json
import re
from collections import Counter
from dataclasses import dataclass

from DatabaseUtils.DatabaseManager import QueryStatus

from sqlglot import parse
from sqlglot.expressions import Expression


SOURCE_NAMESPACE = "source"
GOLD_NAMESPACE = "gold_target"
PREDICTED_NAMESPACE = "target"

@dataclass
class Query:
    ast: Expression
    ignore_reason: str = None
    execution_status: QueryStatus = None
    execution_error: str = None

class PromptDetails:
    def __init__(self, source_cluster: list[str], target_cluster: list[str],
                 extraction_errors: list[str], queries: list[Query]):
        self.source_cluster: list[str] = source_cluster
        self.target_cluster: list[str] = target_cluster
        self.gold_rule_id: str = None
        self.extraction_errors: list[str] = extraction_errors
        self.queries: list[Query] = queries

    def get_valid_queries(self):
        if self.queries is None:
            return []

        try:
            return [q for q in self.queries if q.ignore_reason is None and q.execution_error is None]
        except Exception as e:
            print(f"{e}")

    def get_sql_script(self):
        return "\n\n".join([q.ast.sql() for q in self.get_valid_queries()])


class OverlapResults:

    @dataclass
    class OverlapCount:
        fn: int
        fp: int
        tp: int

    @dataclass
    class OverlapData:
        fn_rows: list[list]
        fp_rows: list[list]
        fn_cols: dict[str, Counter[str]]
        fp_cols: dict[str, Counter[str]]

    def __init__(self):
        self.overlap_count = self.OverlapCount(fn=0, fp=0, tp=0)
        self.overlap_data = self.OverlapData(fn_rows=[], fp_rows=[], fn_cols={}, fp_cols={})


class Results:
    def __init__(self):
        self.table_overlap: dict[str, OverlapResults] = None
        self.join_overlap: dict[str, OverlapResults] = None


class Experiment:

    code_snippet_pattern = re.compile(r"```(.*?)```", flags=re.DOTALL)

    def __init__(self, filepath: str):

        self.mapping_results = Results()
        self.rule_results_by_id: dict[str, Results] = {}

        # Load experiment JSON file and extract any relevant stuff from it into instance variables.
        with open(filepath, 'r') as f:
            json_file = json.load(f)
            self.prompt_config = json_file["prompt_config"]
            self.model_config = json_file["model_config"]

        # Convert JSON prompt data to a list of PromptDetails objects
        self.prompt_details: list[PromptDetails] = []
        for prompt_details in json_file["prompts"]:

            # Extract queries
            extract_errors, parsed_queries = self._extract_and_parse_queries(prompt_details["output"]["response"][0])

            self.prompt_details.append(
                PromptDetails(
                    source_cluster=prompt_details["cluster"]["source"],
                    target_cluster=prompt_details["cluster"]["target"],
                    extraction_errors=extract_errors,
                    queries=parsed_queries))


    def _extract_and_parse_queries(self, full_response: str):

        extract_errs = []

        # Extract the code snippet
        all_snippets = self.code_snippet_pattern.findall(full_response)

        if len(all_snippets) == 0:
            extract_errs.append("No code snippet found.")
            return extract_errs, None
        elif len(all_snippets) > 1:
            extract_errs.append("Multiple code snippets found; only taking the last.")

        snippet: str = all_snippets[-1]
        language = snippet[:3].lower()
        code = snippet[3:]
        if language.lower() != "sql":
            extract_errs.append(f"Language unknown ({language}).")

        # Extract queries from snippet
        try:
            extracted_queries = parse(code)
            if extracted_queries[0] is None:
                extract_errs.append("No SQL statements found in code snippet.")
                return extract_errs, None
        except Exception as e:
            extract_errs.append(str(e))
            return extract_errs, None

        return extract_errs, [Query(ast=q) for q in extracted_queries]