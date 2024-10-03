import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
df = pd.read_csv("./results_reports.csv")

faq_data = df[df['intent_category'] == 'faq']
out_of_scope =  df[df['intent_category'] == 'out_of_scope']


def plot_histograms(aspect, extractive, generative, rs, output_file):
    plt.figure(figsize=(12, 6))

    # Plot Extractive QA
    plt.subplot(1, 3, 1)
    sns.histplot(extractive, kde=True)
    plt.title(f'Extractive QA {aspect}')

    # Plot Generative QA
    plt.subplot(1, 3, 2)
    sns.histplot(generative, kde=True)
    plt.title(f'Generative QA {aspect}')

    # Plot Response Selector
    plt.subplot(1, 3, 3)
    sns.histplot(rs, kde=True)
    plt.title(f'Response Selector {aspect}')

    # Show and save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def q_q_plot (aspect, extractive, generative, rs, output_file):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    stats.probplot(extractive, dist="norm", plot=plt)
    plt.title(f'Extractive QA {aspect} Q-Q Plot')

    plt.subplot(1, 3, 2)
    stats.probplot(generative, dist="norm", plot=plt)
    plt.title(f'Generative QA {aspect} Q-Q Plot')

    plt.subplot(1, 3, 3)
    stats.probplot(rs, dist="norm", plot=plt)
    plt.title(f'Response Selector {aspect} Q-Q Plot')

    plt.savefig(output_file)
    plt.show()


def write_t_test_results(file_name, data, aspect):
    with open(file_name, 'w') as f:
        # Completeness t-tests
        t_stat1, p_val1 = stats.ttest_ind(data["completeness_1"].to_list(), data["completeness_2"].to_list())
        t_stat2, p_val2 = stats.ttest_ind(data["completeness_1"].to_list(), data["completeness_3"].to_list())
        t_stat3, p_val3 = stats.ttest_ind(data["completeness_2"].to_list(), data["completeness_3"].to_list())

        f.write(f"{aspect} Completeness t-test results:\n")
        f.write(f"Extractive vs Generative: t-stat={t_stat1}, p-value={p_val1}\n")
        f.write(f"Extractive vs Response Selector: t-stat={t_stat2}, p-value={p_val2}\n")
        f.write(f"Generative vs Response Selector: t-stat={t_stat3}, p-value={p_val3}\n\n")

        # Factuality t-tests
        t_stat1, p_val1 = stats.ttest_ind(data["factuality_1"].to_list(), data["factuality_2"].to_list())
        t_stat2, p_val2 = stats.ttest_ind(data["factuality_1"].to_list(), data["factuality_3"].to_list())
        t_stat3, p_val3 = stats.ttest_ind(data["factuality_2"].to_list(), data["factuality_3"].to_list())

        f.write(f"{aspect} Factuality t-test results:\n")
        f.write(f"Extractive vs Generative: t-stat={t_stat1}, p-value={p_val1}\n")
        f.write(f"Extractive vs Response Selector: t-stat={t_stat2}, p-value={p_val2}\n")
        f.write(f"Generative vs Response Selector: t-stat={t_stat3}, p-value={p_val3}\n\n")

        # Usefulness t-tests
        t_stat1, p_val1 = stats.ttest_ind(data["usefulness_1"].to_list(), data["usefulness_2"].to_list())
        t_stat2, p_val2 = stats.ttest_ind(data["usefulness_1"].to_list(), data["usefulness_3"].to_list())
        t_stat3, p_val3 = stats.ttest_ind(data["usefulness_2"].to_list(), data["usefulness_3"].to_list())

        f.write(f"{aspect} Usefulness t-test results:\n")
        f.write(f"Extractive vs Generative: t-stat={t_stat1}, p-value={p_val1}\n")
        f.write(f"Extractive vs Response Selector: t-stat={t_stat2}, p-value={p_val2}\n")
        f.write(f"Generative vs Response Selector: t-stat={t_stat3}, p-value={p_val3}\n\n")


# 4. Write FAQ t-test results to 'faq_t_test.txt'
write_t_test_results('faq_t_test.txt', faq_data, 'FAQ')

# 5. Write out-of-scope t-test results to 'out_of_scope_t_test.txt'
write_t_test_results('out_of_scope_t_test.txt', out_of_scope, 'Out of Scope')

print("T-test results have been written to 'faq_t_test.txt' and 'out_of_scope_t_test.txt'")