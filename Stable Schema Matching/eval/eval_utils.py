import re
import json
from collections import namedtuple

from utils.data_builder import get_schemas_for_id



def mean_reciprocal_rank(golden_mapping, predictions):
    golden_set = set(tuple(pair) for pair in golden_mapping)
    rr_list = []

    for target_attr, ranked_preds in predictions.items():
        for rank, pair in enumerate(ranked_preds, start=1):
            if tuple(pair) in golden_set:
                rr_list.append(1.0 / rank)
                break
        else:
            rr_list.append(0.0)  # No correct match found

    return sum(rr_list) / len(rr_list)



def validate_MM_candidates(answer, options):
    if "Refined String List:" in answer:
        refined_list = answer.split("Refined String List:")[1]
    else:
        refined_list = answer
    if not ('[' in refined_list and ']' in refined_list):
        refined_list = '[' + refined_list + ']'
    try:
        result = json.loads(refined_list.replace("'",'"'))
        if isinstance(result, list):
            if len(result) > 5:
                return None
            for attr in result:
                if attr not in options:
                    # print("attr not in options, ", attr)
                    return None
            return result

        else:
            # print("Not a list")
            return None
    except json.JSONDecodeError:
        # print("Invalid list format ", refined_list)
        return None

def _extract_create_view_commands(text):
    """
    Extracts all CREATE VIEW commands from the provided text. Assumes SQLite syntax (see: https://www.sqlite.org/lang_createview.html).

    :param text:
    :return:
    """
    # Regex pattern to match CREATE VIEW commands
    pattern = r'CREATE\s+VIEW.*?;'
    create_view_commands = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    return create_view_commands


def _remove_nulls(old_list):
    '''
           Removes any (null, target_col) alignment pairs. These really aren't an alignment for target_col, these are
           the absence of an alignment.
    '''
    new_list = []
    for source_col, target_col in old_list:
        if source_col != 'null':
            new_list.append((source_col, target_col))
    return new_list


def _parse_view_command(raw_view_command):
    # See https://www.sqlite.org/lang_createview.html for details on which parts of the statement these are meant to parse
    pattern = re.compile(r'''
                                    CREATE\s+VIEW\s+(?P<view_name>[`\"']?[%\w]+[`\"']?)\s+     # Named group for view-name
                                    (?:\((?P<view_columns>[%\w`\"',\s]+)\)\s+)?                # Group for optional column-name
                                    AS\s+(?P<select_statement>SELECT\s+           # Group for full SELECT (first "SELECT" to ";")
                                         (?P<select_cols>(?:[%\w`\"']+\.)?[%\w`\"']+(\s+AS\s+[%\w`\"']+|\s+[%\w`\"']+)?
                          (?:,\s*(?:[%\w`\"']+\.)?[%\w`\"']+(\s+AS\s+[%\w`\"']+|\s+[%\w`\"']+)?)*) 
        \s+FROM\s+(?P<from_table>[`\"']?[%\w]+[`\"']?).*;)  # Group for result-col and from
                                    ''', re.DOTALL | re.VERBOSE | re.IGNORECASE)
    PARSED_VIEW = namedtuple("ParsedView",
                             ["view_name", "view_cols", "select_statement", "select_cols", "from_table"])

    # Split CREATE VIEW clause from SELECT
    match = pattern.search(raw_view_command)

    if match is None:
        return None, f"match is None."
    if match.group('view_name') is None:
        return None, f"view_name is None."
    if match.group('select_statement') is None:
        return None, f"select_statement is None."
    if match.group('select_cols') is None:
        return None, f"select_cols is None."
    if match.group('from_table') is None:
        return None, f"from_table is None."

    view_name = match.group('view_name').strip('`')
    view_cols = match.group('view_columns')
    select_statement = match.group('select_statement')
    select_cols = match.group('select_cols')
    from_table = match.group('from_table')

    # Strip SELECT clause columns of aliases and quotes. SPIDER doesn't like these.
    select_cols_cleaned = []
    for col in select_cols.split(','):
        col = re.sub(r"'|\"|`", "", col.strip())
        if ' ' in col:
            col = col[:col.index(' ')]
        select_cols_cleaned.append(col)
    select_cols_cleaned = ', '.join(select_cols_cleaned)
    select_statement = re.sub(select_cols, select_cols_cleaned, select_statement)

    return PARSED_VIEW(view_name, view_cols, select_statement, select_cols, from_table), None


def _get_view_column_alignments(parsed_view):
    view_cols = parsed_view.view_cols
    select_cols = parsed_view.select_cols

    view_column_mappings = []
    if view_cols is not None:
        split_view_cols = [col.strip().lower() for col in view_cols.split(',')]
    else:
        split_view_cols = None

    split_select_cols = [col.strip().lower() for col in select_cols.split(',')]

    for idx, select_col in enumerate(split_select_cols):
        source_col_name = None
        view_col_name = None

        #  Reassess this if we start using this script for datasets with multiple source tables involved in a single prompt.
        #  Removes the "source_table." prefix from columns--this is implied for out valentine dataset.
        if 'source_table.' == select_col[:13]:
            select_col = select_col[13:]

        if ' ' in select_col or ' as ' in select_col:
            column_and_alias = re.split(' as | ', select_col)
            assert len(column_and_alias) == 2
            source_col_name = column_and_alias[0]
            view_col_name = column_and_alias[1]
        elif split_view_cols is not None and len(split_view_cols) > idx and select_col != split_view_cols[idx]:
            source_col_name = select_col
            view_col_name = split_view_cols[idx]

        # Statement maps source column to a different column name in view
        if source_col_name is not None and view_col_name is not None:
            view_column_mappings.append((source_col_name, view_col_name))
        else:  # Statement maps source column to the same column name in view
            view_column_mappings.append((select_col, select_col))

    return view_column_mappings


def parse_alignments_from_view(raw_text):
    create_views = _extract_create_view_commands(raw_text)

    errors = []
    if len(create_views) == 0:
        errors.append(f'No CREATE VIEWs found')
    else:
        predicted_raw = create_views[0]  # Just take first view found
        predicted_parsed, err = _parse_view_command(predicted_raw)
        if predicted_parsed is None:
            errors.append(f'SQL not parsable: {err}')
        else:
            return errors, _get_view_column_alignments(predicted_parsed)
    return errors, None


def _extract_JSON_yes_no(text):
    pattern = r'({[^\}]*yes[^\}]*no[^\}]*})'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    json_strings = [match.replace('{{', '{').replace('}}', '}').replace("'", '"') for match in matches]

    return json_strings


def _extract_JSON(text):
    pattern = r'({[^\}]*matches[^\}]*})'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    json_strings = [match.replace('{{', '{').replace('}}', '}').replace("'", '"') for match in matches]
    return json_strings


def _extract_TaDa_JSON(text):
    pattern = r'\{\s*"yes"\s*:\s*\[.*?\],\s*"no"\s*:\s*\[.*?\],\s*"unknown"\s*:\s*\[.*?\]\s*\}'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    json_strings = [match.replace('{{', '{').replace('}}', '}').replace("'", '"') for match in matches]
    return json_strings


def _get_json_alignments_yes_no(json_dict):
    return json_dict['yes']


# def _parse_JSON(json_dict):
#     def _format_pairs(list_of_pairs):
#         formatted_pairs = []
#         for pair in list_of_pairs:
#             split_pair = pair.split(',')
#             # formatted_pairs.append((split_pair[0].strip().lower(), split_pair[1].strip().lower()))
#
#             # Clean and format both elements of the pair
#             formatted_first = split_pair[0].strip().strip('<').strip('>').lower().replace('source_table.','')
#             formatted_second = split_pair[1].strip().strip('<').strip('>').lower().replace('target_table.','')
#             formatted_pairs.append((formatted_first, formatted_second))
#
#         return formatted_pairs
#
#     errors = []
#     try:
#         json_dict['matches'] = _format_pairs(json_dict['matches'])
#         # json_dict['yes'] = _format_pairs(json_dict['yes'])
#     except:
#         # errors.append('Formatting issue in "Yes" pairs')
#         errors.append('Formatting issue in "matches" pairs')
#
#     return json_dict, errors


def column_in_schema(column, schema):
    """Check if the column exists in the given schema."""
    result = any(col['name'].lower() == column.lower() for col in schema['columns'])
    return result


def _parse_JSON(json_dict, source_schema, target_schema, nth_attr=0, attr=None):
    def _format_pairs(list_of_pairs):
        formatted_pairs = []
        for pair in list_of_pairs:
            split_pair = pair.split(',')
            # Clean and format both elements of the pair
            formatted_first = split_pair[0].strip().strip('<').strip('>').lower().replace('source_table.', '')
            formatted_second = split_pair[1].strip().strip('<').strip('>').lower().replace('target_table.', '')
            formatted_pairs.append((formatted_first, formatted_second))

        return formatted_pairs

    errors = []
    mapping_errors = []
    same_schema_count = 0
    same_attribute_count = 0
    invalid_attribute_count = 0

    valid_attribute_pairs = []

    try:
        mappings = _format_pairs(json_dict['matches'])

        for formatted_first, formatted_second in mappings:
            # Check if both are from the source schema or both from the target schema
            first_in_source = column_in_schema(formatted_first, source_schema)
            second_in_source = column_in_schema(formatted_second, source_schema)
            first_in_target = column_in_schema(formatted_first, target_schema)
            second_in_target = column_in_schema(formatted_second, target_schema)

            if first_in_source and second_in_target:
                if nth_attr == 1 and formatted_first != attr:
                    mapping_errors.append(
                        f"'one-to-n formatting Error: {attr} not in {(formatted_first, formatted_second)}'")
                    continue
                if nth_attr == 2 and formatted_second != attr:
                    mapping_errors.append(
                        f"'n-to-one formatting Error: {attr} not in {(formatted_first, formatted_second)}'")
                    continue

                valid_attribute_pairs.append((formatted_first, formatted_second))
                continue

            if not first_in_source and not first_in_target:
                invalid_attribute_count += 1
                mapping_errors.append(
                    f"Source column '{formatted_first}' in '{(formatted_first, formatted_second)}' doesn't exist in source or target schema.")
            if not second_in_source and not second_in_target:
                invalid_attribute_count += 1
                mapping_errors.append(
                    f"Target column '{formatted_second}' in '{(formatted_first, formatted_second)}' doesn't exist in source or target schema.")

            mapping_err = ''
            # Check if both columns are from the source or both from the target
            if first_in_source and second_in_source and not second_in_target:
                same_schema_count += 1
                mapping_err += f"Target column '{formatted_second}' in '{(formatted_first, formatted_second)}' exist in source Only."

                if formatted_first == formatted_second:
                    same_attribute_count += 1
                    mapping_err += "Its same attribute pair."

                mapping_errors.append(mapping_err)

            # Check if both columns are from the source or both from the target
            if first_in_target and second_in_target and not first_in_source:
                same_schema_count += 1
                mapping_err += f"Target column '{formatted_second}' in '{(formatted_first, formatted_second)}' exist in source Only."

                if formatted_first == formatted_second:
                    same_attribute_count += 1
                    mapping_err += "Its same attribute pair."

                mapping_errors.append(mapping_err)


    except Exception as e:
        errors.append(f"Formatting issue in 'matches' pairs: {str(e)}")

    if len(valid_attribute_pairs) == 0:
        json_alignments = None
    else:
        # json_dict['matches'] = valid_attribute_pairs
        json_alignments = valid_attribute_pairs
    stats = {'mapping_errors': mapping_errors,
             'mapping_errors_cnt': len(mapping_errors),
             'same_schema_count': same_schema_count,
             'same_attribute_count': same_attribute_count,
             'invalid_attribute_count': invalid_attribute_count}

    return json_alignments, errors, stats
    # return json_dict, errors, stats


def _parse_TaDa_JSON(json_dict, source_schema, target_schema, nth_attr=0, attr=None):
    def _format_pairs(query, list_attrs):
        formatted_pairs = []
        for pred_match in list_attrs:
            # Clean and format both elements of the pair
            pred_match = pred_match.strip().strip('<').strip('>').lower()
            formatted_pairs.append((pred_match, query))
        # print(formatted_pairs)
        return formatted_pairs

    errors = []
    mapping_errors = []
    same_schema_count = -1
    same_attribute_count = -1
    invalid_attribute_count = 0

    valid_attribute_pairs = {"yes": [], "no": []}

    try:
        # print("\n\n")
        # print(json_dict)
        yes_list = _format_pairs(attr, json_dict['yes'])
        no_list = _format_pairs(attr, json_dict['no'])

        for formatted_first, formatted_second in yes_list:
            first_in_source = column_in_schema(formatted_first, source_schema)

            if not first_in_source:
                invalid_attribute_count += 1
                # print(
                #     f"Source column '{formatted_first}' in '{(formatted_first, formatted_second)}' doesn't exist in source")

                mapping_errors.append(
                    f"Source column '{formatted_first}' in '{(formatted_first, formatted_second)}' doesn't exist in source")
            else:
                valid_attribute_pairs["yes"].append((formatted_first, formatted_second))

        for formatted_first, formatted_second in no_list:
            first_in_source = column_in_schema(formatted_first, source_schema)

            if not first_in_source:
                invalid_attribute_count += 1
                # print(
                #     f"Source column '{formatted_first}' in '{(formatted_first, formatted_second)}' doesn't exist in source")

                mapping_errors.append(
                    f"Source column '{formatted_first}' in '{(formatted_first, formatted_second)}' doesn't exist in source")
            else:
                valid_attribute_pairs["no"].append((formatted_first, formatted_second))



    except Exception as e:
        errors.append(f"Formatting issue in 'matches' pairs: {str(e)}")

    if len(valid_attribute_pairs) == 0:
        json_alignments = None
    else:
        json_alignments = valid_attribute_pairs

    # print('\njson_alignments')
    # print(json_alignments)
    stats = {'mapping_errors': mapping_errors,
             'mapping_errors_cnt': len(mapping_errors),
             'same_schema_count': same_schema_count,
             'same_attribute_count': same_attribute_count,
             'invalid_attribute_count': invalid_attribute_count}

    return json_alignments, errors, stats


# def parse_alignments_from_JSON(raw_text):
#     # print('raw_text')
#     # print(raw_text)
#     json_strings = _extract_JSON(raw_text)
#     # print('\n\njson_strings')
#     # print(json_strings)
#     errors = []
#     if len(json_strings) == 0:
#         errors.append(f'No valid JSON found')
#     else:
#         predicted_raw = json_strings[0]  # Just take first JSON dict found
#         # print(predicted_raw)
#         try:
#             json_dict = json.loads(predicted_raw)
#             predicted_parsed, errors = _parse_JSON(json_dict)
#             return errors, _get_json_alignments(predicted_parsed)
#         except json.JSONDecodeError:
#             errors.append("No valid JSON found")
#
#     return errors, None

def parse_alignments_from_JSON(raw_text, source_schema, target_schema, nth_attr=0, attr=None, isTaDa=False):
    if isTaDa:
        json_strings = _extract_TaDa_JSON(raw_text)
    else:
        json_strings = _extract_JSON(raw_text)
    errors = []

    valid_json_found = False

    # Iterate over each JSON string extracted
    for json_str in json_strings:
        try:
            # Try to load the JSON string
            json_dict = json.loads(json_str)
            # If successful, process it
            if isTaDa:
                json_alignments, errors, stats = _parse_TaDa_JSON(json_dict, source_schema, target_schema, nth_attr,
                                                                  attr)
            else:
                json_alignments, errors, stats = _parse_JSON(json_dict, source_schema, target_schema, nth_attr, attr)
            valid_json_found = True
            return errors, json_alignments, stats
        except json.JSONDecodeError:
            # Continue to the next JSON string
            continue

    # If no valid JSON was found
    if not valid_json_found:
        errors.append(f'No valid JSON found, json_strings: {json_strings}')

    return errors, None, None


'''
    Removes any (null, target_col) alignment pairs. These really aren't an alignment for target_col, these are 
    the absence of an alignment.
'''


def remove_nulls(old_list):
    new_list = []
    for source_col, target_col in old_list:
        if source_col != 'null':
            new_list.append((source_col, target_col))
    return new_list

def get_tn(test_case, target_schema):
    source_cols = set([x["name"].lower() for x in target_schema["columns"]])
    aligned = set([x[1].lower() for x in test_case['gold_mapping']])
    no_alignment = source_cols - aligned
    predicted_mapping_key = 'predicted_mapping'
    if predicted_mapping_key not in test_case:
        predicted_mapping_key = 'predicted_mappings'
    mapped_cols = set([x[1].lower() for x in test_case[predicted_mapping_key]])
    return len([x for x in no_alignment if x not in mapped_cols])


def compute_prf1e(tp, fn, fp, test_case, dataset_name):
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if recall + precision == 0:
        f1 = 0
    else:
        f1 = (2. * recall * precision) / (recall + precision)

    source_schema, target_schema = get_schemas_for_id(test_case, dataset_name)
    N_ = len(target_schema["columns"])
    tn = get_tn(test_case, target_schema)
    accuracy = (tp+tn) / N_
    accuracy2 = (tp+tn) / (tp+tn+fp+fn)


    len_gold = len(test_case["gold_mapping"])
    e = (3 * fn + fp) / len_gold

    return precision, recall, f1, accuracy, accuracy2, e


def get_coverage_ratio(view):
    errors, align_false_true = parse_alignments_from_view(view)
    allign_true = remove_nulls(align_false_true)
    return len(allign_true) / len(align_false_true)
