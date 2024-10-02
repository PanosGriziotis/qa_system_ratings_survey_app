import pandas as pd
import matplotlib.pyplot as plt

def load_result_file(filepath):
    return pd.read_csv(filepath)


def plot_box_plots(df, category):
    # Filter data for the given category
    category_df = df[df['intent_category'] == category]

    # Extract ratings for each Answer and rating aspect
    aspects = ['completeness', 'factuality', 'usefulness']
    answers = ['answer_1', 'answer_2', 'answer_3']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True) 
    
    box_colors = ['skyblue', 'orange', 'green']  # Colors for the box plots

    for i, aspect in enumerate(aspects):
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        colors = []  # This list should reset for each aspect

        for j, answer in enumerate(answers):
            answer_idx = answer.split ("_")[1]
            # Collect data for available answers
            aspect_column = f'{aspect}_{answer_idx}'
            if aspect_column in category_df.columns:
                valid_ratings = category_df[aspect_column].dropna().values
                if len(valid_ratings) > 0:
                    data_to_plot.append(valid_ratings)
                    colors.append(box_colors[j])  # Assign color for the corresponding answer
                    if answer == 'answer_1':
                        labels.append('Extractive QA')
                    elif answer == 'answer_2':
                        labels.append('Generative QA')
                    elif answer == 'answer_3' and category == 'faq':  # Only for faq
                        labels.append('Response Selector')

        # Plot box plot
        box = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Set colors for each box
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        if aspect=="usefulness":
            title = "Overall Usefulness"
        else:
            title = aspect.capitalize()
        axes[i].set_title(f'{title}')
        axes[i].set_ylim(0.8, 5.2)  # Adjusted to create space above the highest rating (5)
        axes[i].set_ylabel('Ratings')

    plt.savefig(f"{category}_plot.png")


df = load_result_file("./results.csv")
# Plot for 'out_of_scope' category
plot_box_plots(df, 'out_of_scope')

# Plot for 'faq' category
plot_box_plots(df, 'faq')