\"""Validators for innovations (accuracy, performance)\"""

def validate_accuracy(predictions, ground_truth):
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(ground_truth) if ground_truth else 0.0

def validate_performance(runtime, threshold=1.0):
    return runtime <= threshold
