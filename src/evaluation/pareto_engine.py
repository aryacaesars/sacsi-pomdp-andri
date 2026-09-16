"""Pareto analysis helper."""

def dominates(a, b):
    return a[0] >= b[0] and a[1] <= b[1] and a != b
