import os
import pandas as pd
import random
import numpy as np


def get_dataset_name(path):
    if not os.path.isdir(path): # Path is to a results file
        path, _ = os.path.split(path)

    path, _ = os.path.split(path)
    _, dataset_dir = os.path.split(path)

    return dataset_dir

def get_dataset_name_from_id(id):
    if '/' in id:
        return 'valentine'
    elif '|' in id:
        if id.count('|') == 2:
            return 'bird'
        elif id.count('|') == 1 and 'synthea' in id:
            return 'synthea'
        elif id.count('|') == 1 and 'gdc' in id:
            return 'gdc'
        elif id.count('|') == 1:
            return 'ehr'

def parse_gdc_id_parts(id_string):
    source_id, target_id = id_string.split('|')
    source_db, source_table = source_id.split(':')
    target_db, target_table = target_id.split(':')
    return source_db, source_table, target_db, target_table

def parse_ehr_id_parts(id_string):
    source_id, target_id = id_string.split('|')
    source_db, source_table = source_id.split(':')
    target_db, target_table = target_id.split(':')
    return source_db, source_table, target_db, target_table

def parse_bird_id_parts(id_string):
    domain, source_id, target_id = id_string.split('|')
    source_db, source_table = source_id.split(':')
    target_db, target_table = target_id.split(':')
    return domain, source_db, source_table, target_db, target_table

def parse_synthea_id_parts(id_string):
    source_id, target_id = id_string.split('|')
    source_db, source_table = source_id.split(':')
    target_db, target_table = target_id.split(':')
    return source_db, source_table, target_db, target_table

class InstanceDataHandler:
    def __init__(self, seed, n_col_example=None, data_instance_selector='random'):
        self.n_col_example = n_col_example
        self.data_instance_selector = data_instance_selector
        self.random_seed = seed  # Store the seed
        self.rng = None     # For standard random package
        self.np_rng = None  # For numpy's random package

    def get_column_order_and_types_from_sql_schema(self, schema):
        start = schema.find('(') + 1
        end = schema.rfind(')')
        columns = schema[start:end].split(',')

        column_types = {}

        column_names = []

        for col in columns:
            # Remove any leading/trailing whitespace and split by whitespace
            parts = col.strip().split()
            column_name = parts[0]  # The first part is the column name
            column_type = parts[1] if len(parts) > 1 else None  # The second part is the type

            column_names.append(column_name)
            if column_type:
                column_types[column_name] = column_type

        return column_names, column_types

    def get_column_order_and_types(self, schema):
        column_types = {}
        column_names = []

        for cols in schema['columns']:
            column_name = cols['name']
            column_type = cols['type']

            column_names.append(column_name)
            column_types[column_name] = column_type

        return column_names, column_types

    # Function to standardize column names
    def standardize_columns(self, columns):
        return [col.strip().replace(' ', '')
                .replace('(', '')
                .replace(')', '')
                .replace('-', '')
                .replace('_', '')
                for col in columns]

    def _get_valentine_dataframes(self, example):
        path = example["id"]
        full_path = os.path.join('data/valentine', path)

        source_file = os.path.join(full_path, f"{os.path.basename(path)}_source.csv")
        target_file = os.path.join(full_path, f"{os.path.basename(path)}_target.csv")

        try:
            if example['swapped']:
                target_file, source_file = source_file, target_file
        except:
            print('No "swapped" key found. You must be using an old dataset output.')

        source_schema = example["source_schema"]
        target_schema = example["target_schema"]

        source_schema_columns_original, source_column_types = self.get_column_order_and_types(source_schema)
        target_schema_columns_original, target_column_types = self.get_column_order_and_types(target_schema)
        source_df = pd.read_csv(source_file, dtype=str)
        target_df = pd.read_csv(target_file, dtype=str)

        common_source_columns = [col for col in source_schema_columns_original if
                                 col in source_df.columns]

        common_target_columns = [col for col in target_schema_columns_original if
                                 col in target_df.columns]

        source_df = source_df[common_source_columns]
        source_df.columns = source_schema_columns_original

        target_df = target_df[common_target_columns]
        target_df.columns = target_schema_columns_original
        return source_df, target_df, source_column_types, target_column_types

    def _map_ehs_table(self, df, db, table):

        # Remove row_id -- simply acts as an arbitrary PK (even when other PKs exist), and is not considered part of the schema
        if db == 'mimic-iii' and 'row_id' in df.columns:
            df = df.drop('row_id', axis='columns')

        # Different version of OMOP is used for csvs, so some difference in columns from schema file. Map them here.
        if db == 'omop':
            if table == 'VISIT_DETAIL' or table == 'VISIT_OCCURRENCE':
                df = df.rename(columns={'admitting_source_value': 'admitted_from_source_value',
                                                  'admitting_source_concept_id': 'admitted_from_concept_id'})


        return df

    def _get_ehr_dataframes(self, example):
        source_db, source_table, target_db, target_table = parse_ehr_id_parts(example["id"])

        try:
            if example['swapped']:
                target_db, source_db = source_db, target_db
                target_table, source_table = source_table, target_table
        except:
            print('No "swapped" key found. You must be using an old dataset output.')

        full_path = 'data/ehr/datasources'

        source_file_upper = os.path.join(full_path, f'{source_db}/data/{source_table.upper()}.csv')
        source_file_lower = os.path.join(full_path, f'{source_db}/data/{source_table.lower()}.csv')
        if os.path.exists(source_file_upper):
            source_file = source_file_upper
        elif os.path.exists(source_file_lower):
            source_file = source_file_lower
        else:
            print(f'{source_file_upper} not found')

        target_file_upper = os.path.join(full_path, f'{target_db}/data/{target_table.upper()}.csv')
        target_file_lower = os.path.join(full_path, f'{target_db}/data/{target_table.lower()}.csv')
        if os.path.exists(target_file_upper):
            target_file = target_file_upper
        elif os.path.exists(target_file_lower):
            target_file = target_file_lower
        else:
            print(f'{target_file_upper} not found')

        source_schema = example["source_schema"]
        target_schema = example["target_schema"]

        source_schema_columns, source_column_types = self.get_column_order_and_types(source_schema)
        target_schema_columns, target_column_types = self.get_column_order_and_types(target_schema)
        source_df = pd.read_csv(source_file, dtype=str)
        target_df = pd.read_csv(target_file, dtype=str)

        source_df = self._map_ehs_table(source_df, source_db, source_table)
        target_df = self._map_ehs_table(target_df, target_db, target_table)

        # Add back columns from schema (these may have been filtered out if data for them did not exist in the csv)
        missing_source_cols = set(source_schema_columns) - set(source_df.columns)
        missing_target_cols = set(target_schema_columns) - set(target_df.columns)
        if len(missing_source_cols) > 0:
            print(f'{source_db} missing columns in {source_table}.csv: {missing_source_cols} (cols added and set to None)')
        if len(missing_target_cols) > 0:
            print(f'{target_db} missing columns in {target_table}.csv: {missing_target_cols} (cols added and set to None)')
        source_df[list(missing_source_cols)] = None
        target_df[list(missing_target_cols)] = None

        if len(source_df) == 0:
            print(f'{source_db}/{source_table}.csv is empty')
        if len(target_df) == 0:
            print(f'{target_db}/{target_table}.csv is empty')

        assert len(source_df.columns) == len(
            source_schema_columns), f"Schema and CSV #columns do not match: {source_db}/{source_table}"
        source_df = source_df[source_schema_columns]
        source_df.columns = source_schema_columns

        assert len(target_df.columns) == len(
            target_schema_columns), f"Schema and CSV #columns do not match: {target_db}/{target_table}"
        target_df = target_df[target_schema_columns]
        target_df.columns = target_schema_columns

        return source_df, target_df, source_column_types, target_column_types

    def _get_bird_dataframes(self, example):
        domain, source_db, source_table, target_db, target_table = parse_bird_id_parts(example["id"])

        try:
            if example['swapped']:
                target_db, source_db = source_db, target_db
                target_table, source_table = source_table, target_table
        except:
            print('No "swapped" key found. You must be using an old dataset output.')

        full_path = f'data/bird/{domain}'

        source_file = os.path.join(full_path, f'{source_db}/data/{source_table}.csv')
        if not os.path.exists(source_file):
            print(f'{source_file} not found')
        target_file = os.path.join(full_path, f'{target_db}/data/{target_table}.csv')
        if not os.path.exists(target_file):
            print(f'{target_file} not found')

        source_schema = example["source_schema"]
        target_schema = example["target_schema"]

        source_schema_columns, source_column_types = self.get_column_order_and_types(source_schema)
        target_schema_columns, target_column_types = self.get_column_order_and_types(target_schema)
        source_df = pd.read_csv(source_file, dtype=str)
        target_df = pd.read_csv(target_file, dtype=str)

        # Add back columns from schema (these may have been filtered out if data for them did not exist in the csv)
        missing_source_cols = set(source_schema_columns) - set(source_df.columns)
        missing_target_cols = set(target_schema_columns) - set(target_df.columns)
        # print('-' * 50)
        if len(missing_source_cols) > 0:
            print(f'{source_db} missing columns in {source_table}.csv: {missing_source_cols} (cols added and set to None)')
        if len(missing_target_cols) > 0:
            print(f'{target_db} missing columns in {target_table}.csv: {missing_target_cols} (cols added and set to None)')
        # print('-'*50)
        source_df[list(missing_source_cols)] = None
        target_df[list(missing_target_cols)] = None

        if len(source_df) == 0:
            print(f'{source_db}/{source_table}.csv is empty')
        if len(target_df) == 0:
            print(f'{target_db}/{target_table}.csv is empty')

        assert len(source_df.columns) == len(
            source_schema_columns), f"Schema and CSV #columns do not match: {source_db}/{source_table}"
        source_df = source_df[source_schema_columns]
        source_df.columns = source_schema_columns

        assert len(target_df.columns) == len(
            target_schema_columns), f"Schema and CSV #columns do not match: {target_db}/{target_table}"
        target_df = target_df[target_schema_columns]
        target_df.columns = target_schema_columns

        return source_df, target_df, source_column_types, target_column_types

    def _map_synthea_table(self, df, db, table):

        ## Remove row_id -- simply acts as an arbitrary PK (even when other PKs exist), and is not considered part of the schema
        # if db == 'synthea' and 'row_id' in df.columns:
        #     df = df.drop('row_id', axis='columns')

        # Different version of OMOP is used for csvs, so some difference in columns from schema file. Map them here.
        if db == 'omop':
            if table == 'VISIT_DETAIL' or table == 'VISIT_OCCURRENCE':
                df = df.rename(columns={'admitting_source_value': 'admitted_from_source_value',
                                        'admitting_source_concept_id': 'admitted_from_concept_id'})

        return df

    def _get_synthea_dataframes(self, example):

        source_db, source_table, target_db, target_table = parse_synthea_id_parts(example["id"])

        full_path = 'data/synthea/datasources'

        source_file = os.path.join(full_path, f'{source_db}/data/{source_table.lower()}.csv')
        target_file = os.path.join(full_path, f'{target_db}/data/{target_table.lower()}.csv')

        source_schema = example["source_schema"]
        target_schema = example["target_schema"]

        source_schema_columns, source_column_types = self.get_column_order_and_types(source_schema)
        target_schema_columns, target_column_types = self.get_column_order_and_types(target_schema)
        source_df = pd.read_csv(source_file, dtype=str)
        source_df.columns = source_df.columns.str.lower()
        target_df = pd.read_csv(target_file, dtype=str)
        target_df.columns = target_df.columns.str.lower()


        source_df = self._map_synthea_table(source_df, source_db, source_table)
        target_df = self._map_synthea_table(target_df, target_db, target_table)

        missing_source_cols = set(source_schema_columns) - set(source_df.columns)
        missing_target_cols = set(target_schema_columns) - set(target_df.columns)

        if len(missing_source_cols) > 0:
            print(
                f'{source_db} missing columns in {source_table}.csv: {missing_source_cols} (cols added and set to None)')
        if len(missing_target_cols) > 0:
            print(
                f'{target_db} missing columns in {target_table}.csv: {missing_target_cols} (cols added and set to None)')
        source_df[list(missing_source_cols)] = None
        target_df[list(missing_target_cols)] = None


        if len(source_df) == 0:
            print(f'{source_db}/{source_table}.csv is empty')
        if len(target_df) == 0:
            print(f'{target_db}/{target_table}.csv is empty')


        source_df = source_df[source_schema_columns]
        source_df.columns = source_schema_columns
        assert len(source_df.columns) == len(
            source_schema_columns), f"Schema and CSV #columns do not match: {source_db}/{source_table}"

        target_df = target_df[target_schema_columns]
        target_df.columns = target_schema_columns
        assert len(target_df.columns) == len(
            target_schema_columns), f"Schema and CSV #columns do not match: {target_db}/{target_table}"


        return source_df, target_df, source_column_types, target_column_types


    def _get_gdc_dataframes(self, example):

        source_db, source_table, target_db, target_table = parse_gdc_id_parts(example["id"])

        full_path = 'data/gdc'

        source_file = os.path.join(full_path, f'{source_db}/{source_table}.csv')
        target_file = os.path.join(full_path, f'{target_db}/{target_table}.csv')

        source_schema = example["source_schema"]
        target_schema = example["target_schema"]

        source_schema_columns, source_column_types = self.get_column_order_and_types(source_schema)
        target_schema_columns, target_column_types = self.get_column_order_and_types(target_schema)
        source_df = pd.read_csv(source_file, dtype=str)
        target_df = pd.read_csv(target_file, dtype=str)

        missing_source_cols = set(source_schema_columns) - set(source_df.columns)
        missing_target_cols = set(target_schema_columns) - set(target_df.columns)

        if len(missing_source_cols) > 0:
            print(
                f'{source_db} missing columns in {source_table}.csv: {missing_source_cols} (cols added and set to None)')
        if len(missing_target_cols) > 0:
            print(
                f'{target_db} missing columns in {target_table}.csv: {missing_target_cols} (cols added and set to None)')
        source_df[list(missing_source_cols)] = None
        target_df[list(missing_target_cols)] = None


        if len(source_df) == 0:
            print(f'{source_db}/{source_table}.csv is empty')
        if len(target_df) == 0:
            print(f'{target_db}/{target_table}.csv is empty')


        source_df = source_df[source_schema_columns]
        source_df.columns = source_schema_columns
        assert len(source_df.columns) == len(
            source_schema_columns), f"Schema and CSV #columns do not match: {source_db}/{source_table}"

        target_df = target_df[target_schema_columns]
        target_df.columns = target_schema_columns
        assert len(target_df.columns) == len(
            target_schema_columns), f"Schema and CSV #columns do not match: {target_db}/{target_table}"


        return source_df, target_df, source_column_types, target_column_types


    def get_dataframes(self, example):
        dataset_name = get_dataset_name_from_id(example['id'])
        if dataset_name == 'valentine':
            return self._get_valentine_dataframes(example)
        elif dataset_name == 'ehr':
            return self._get_ehr_dataframes(example)
        elif dataset_name == 'bird':
            return self._get_bird_dataframes(example)
        elif dataset_name == 'synthea':
            return self._get_synthea_dataframes(example)
        elif dataset_name == 'gdc':
            return self._get_gdc_dataframes(example)
        else:
            print('Unknown dataset')

    def get_rows(self, example, n_rows):
        # TODO : random function need to be checked
        source_df, target_df, _, _ = self.get_dataframes(example)
        if self.rng is None:
            self.rng = random.Random(self.random_seed)
        # rng = np.random.default_rng(self.random_seed)

        source_random_rows = source_df.sample(n=min(n_rows, len(source_df)), replace=False,
                                              random_state=self.rng).values.tolist()
        target_random_rows = target_df.sample(n=min(n_rows, len(target_df)), replace=False,
                                              random_state=self.rng).values.tolist()

        return source_random_rows, target_random_rows



    def get_weighted_samples(self, df, n):
        """
        This function returns the top `n` distinct values from each column in the dataframe `df`
        using weighted sampling based on the inverse of value frequencies.
        """
        top_n_values = {}
        if self.np_rng is None:
            self.np_rng = np.random.default_rng(self.random_seed)

        for col in df.columns:

            filtered_df_col = [x for x in df[col].tolist() if pd.notna(x)]
            value_counts = pd.Series(filtered_df_col).value_counts(normalize=True)

            if len(value_counts) == 1:
                weights = np.ones(len(filtered_df_col))
            else:
                weights = [1 - value_counts.get(x, 0) for x in filtered_df_col]
                weights = np.array(weights) / np.sum(weights)

            if not np.isclose(np.sum(weights), 1.0):
                weights = weights / np.sum(weights)

            if len(filtered_df_col) < n:
                top_n_values[col] = filtered_df_col
                continue

            top_n_values[col] = self.np_rng.choice(filtered_df_col, size=n, replace=False, p=weights).tolist()

        return top_n_values

    def get_random_samples(self, df, n):
        top_n_values = {}
        if self.np_rng is None:
            self.np_rng = np.random.default_rng(self.random_seed)

        for col in df.columns:

            filtered_df_col = [x for x in df[col].tolist() if pd.notna(x)]

            if len(filtered_df_col) < n:
                top_n_values[col] = filtered_df_col
                continue

            top_n_values[col] = self.np_rng.choice(filtered_df_col, size=n, replace=False).tolist()

        return top_n_values

    def get_unique_random_samples(self, df, n):
        unique_n_values = {}
        if self.np_rng is None:
            self.np_rng = np.random.default_rng(self.random_seed)

        for col in df.columns:
            # Filter out NaN values and get unique values
            filtered_df_col = list(set([x for x in df[col].tolist() if pd.notna(x)]))
            # print(filtered_df_col)

            # Adjust n if the number of unique values is less than n
            if len(filtered_df_col) < n:
                unique_n_values[col] = filtered_df_col
                continue

            unique_n_values[col] = self.np_rng.choice(filtered_df_col, size=n, replace=False).tolist()

        return unique_n_values

    def get_n_distinct_col_values(self, example):
        source_df, target_df, source_column_types, target_column_types = self.get_dataframes(example)

        if self.n_col_example is None:
            n = 0
        else:
            n = self.n_col_example

        if self.data_instance_selector == 'most_frequent':
            # Get n most occurred distinct values for each column in source and target dataframes
            source_top_n_values = {col: source_df[col].value_counts().index[:n].tolist() for col in source_df.columns}
            target_top_n_values = {col: target_df[col].value_counts().index[:n].tolist() for col in target_df.columns}

        elif self.data_instance_selector == 'weighted_sampler':
            # Use the function to get weighted samples for both source and target dataframes
            source_top_n_values = self.get_weighted_samples(source_df, n)
            target_top_n_values = self.get_weighted_samples(target_df, n)

        elif self.data_instance_selector == 'random_unique':
            # Use the function to get weighted samples for both source and target dataframes
            source_top_n_values = self.get_unique_random_samples(source_df, n)
            target_top_n_values = self.get_unique_random_samples(target_df, n)

        else:
            #if self.data_instance_selector == 'random':
            # Use the function to get random samples for both source and target dataframes

            source_top_n_values = self.get_random_samples(source_df, n)
            target_top_n_values = self.get_random_samples(target_df, n)
        return source_top_n_values, target_top_n_values, source_column_types, target_column_types

