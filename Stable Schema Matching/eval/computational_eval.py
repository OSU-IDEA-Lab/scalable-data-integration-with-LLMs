import json
import math
import os
from scipy import stats


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_total_tokens(data):
    token_cnt = sum(entry["prompt_tokens"] for entry in data)
    num_questions = len(data)
    return token_cnt, num_questions


def report_statistics(token_sums, question_counts):
    mean_total = sum(token_sums) / sum(question_counts)
    print(f"Mean tokens per question: {mean_total:.2f}")

    means_per_file = [token_sum / count for token_sum, count in zip(token_sums, question_counts)]

    overall_mean = sum(means_per_file) / len(means_per_file)
    std_error = stats.sem(means_per_file)
    t_value = stats.t.ppf(0.95, df=len(means_per_file) - 1)

    margin_of_error = t_value * std_error

    print(f"Mean tokens per question : {overall_mean:.2f} ± {margin_of_error:.2f}")

def load_generation_time(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data["generation_time"]


def report_generation_time_statistics(generation_times):
    mean_time = sum(generation_times) / len(generation_times)
    std_error = stats.sem(generation_times)
    t_value = stats.t.ppf(0.95, df=len(generation_times) - 1)  # 95% CI
    margin_of_error = t_value * std_error

    mean_time_minutes = mean_time / 60
    margin_of_error_minutes = margin_of_error / 60

    print(f"Mean: {mean_time_minutes:.2f} ± {margin_of_error_minutes:.2f} minutes")

def print_computational_cost(result_files, question_files, seeds, base_dir):
    token_sums = []
    question_counts = []

    for i in range(len(question_files)):
        data = load_json(os.path.join(base_dir, question_files[i]))
        token_sum, num_questions = calculate_total_tokens(data)
        token_sums.append(token_sum)
        question_counts.append(num_questions)
        print(
            f" Seed {seeds[i]}, Total {num_questions} questions, {token_sum} tokens per prompt, {token_sum / num_questions:.2f} tokens per question")

    report_statistics(token_sums, question_counts)

    generation_times = [load_generation_time(os.path.join(base_dir, file_path)) for file_path in result_files]
    report_generation_time_statistics(generation_times)
