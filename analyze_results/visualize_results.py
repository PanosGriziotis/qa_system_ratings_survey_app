import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_result_file(filepath):
    return pd.read_csv(filepath)

# Function to annotate p-values with vertical offsets to prevent overlap
def annotate_p_values(axes, x1, x2, y, p_val, offset=0, line_color='black', text_color='black'):
    """Helper function to annotate p-values on the plot with adjustable offset and line color."""
    h = 0.05 + offset  # Adjust the height dynamically with offset
    line_style = {'lw': 1, 'color': line_color, 'linestyle': '--'}  # More discreet lines
    axes.plot([x1, x1, x2, x2], [y, y+h, y+h, y], **line_style)  # draw the line between boxplots
    # annotate p-value, adjusting color
    axes.text((x1+x2)*0.5, y+h, f'p = {p_val:.2e}', ha='center', va='bottom', color=text_color, fontsize=9)  # smaller font

def plot_box_plots(df, category):
    # Filter data for the given category
    category_df = df[df['intent_category'] == category]

    # Extract ratings for each Answer and rating aspect
    aspects = ['completeness', 'factuality', 'usefulness']
    answers = ['answer_1', 'answer_2', 'answer_3']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)  # Increased figure size
    box_colors = ['skyblue', 'orange', 'green']  # Colors for the box plots
    system_labels = ['Extractive QA', 'Generative QA', 'Response Selector']

    for i, aspect in enumerate(aspects):
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        colors = []  # This list should reset for each aspect

        for j, answer in enumerate(answers):
            answer_idx = answer.split("_")[1]
            aspect_column = f'{aspect}_{answer_idx}'
            if aspect_column in category_df.columns:
                valid_ratings = category_df[aspect_column].dropna().values
                if len(valid_ratings) > 0:
                    data_to_plot.append(valid_ratings)
                    colors.append(box_colors[j])  # Assign color for the corresponding answer
                    if answer == 'answer_1':
                        labels.append(system_labels[0])
                    elif answer == 'answer_2':
                        labels.append(system_labels[1])
                    elif answer == 'answer_3' and category == 'faq':  # Only for FAQ category
                        labels.append(system_labels[2])

        # Plot box plot
        box = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Set colors for each box
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Set title and labels
        title = "Overall Usefulness" if aspect == "usefulness" else aspect.capitalize()
        axes[i].set_title(f'{title}')
        axes[i].set_ylim(0.5, 5.9)  # Adjust y-axis to create more space (lower limit 0.5, upper limit 5.7)
        axes[i].set_ylabel('Ratings')

        # Perform t-tests and annotate p-values
        if len(data_to_plot) >= 2:
            # Extractive vs Generative
            t_stat1, p_val1 = stats.ttest_ind(data_to_plot[0], data_to_plot[1])
            annotate_p_values(axes[i], 1, 2, 5.3, p_val1, offset=0.0, line_color='blue', text_color='blue')

            # Extractive vs Response Selector (if present)
            if len(data_to_plot) == 3:
                t_stat2, p_val2 = stats.ttest_ind(data_to_plot[0], data_to_plot[2])
                t_stat3, p_val3 = stats.ttest_ind(data_to_plot[1], data_to_plot[2])
                annotate_p_values(axes[i], 1, 3, 5.3, p_val2, offset=0.15, line_color='green', text_color='green')
                annotate_p_values(axes[i], 2, 3, 5.1, p_val3, offset=0.3, line_color='red', text_color='red')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout with tighter spacing
    plt.subplots_adjust(top=0.88, bottom=0.12)  # More space on top and bottom
    plt.savefig(f"{category}_plot_with_pvalues_fixed.png")

# Load and plot data
df = load_result_file("./results_reports.csv")
plot_box_plots(df, 'out_of_scope')
plot_box_plots(df, 'faq')