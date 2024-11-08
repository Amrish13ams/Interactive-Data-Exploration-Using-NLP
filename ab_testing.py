
import pandas as pd
from scipy.stats import chi2_contingency

def load_data(file_path):
    return pd.read_csv(file_path)

def ab_test(data):
    contingency_table = pd.crosstab(data['strategy'], data['purchase'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

if __name__ == "__main__":
    data = load_data('data/ab_test_data.csv')
    p_value = ab_test(data)
    print(f"A/B Test p-value: {p_value}")
    if p_value < 0.05:
        print("Significant difference detected between strategies.")
    else:
        print("No significant difference between strategies.")
    