import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loads the dataset
dataset_path = "datasets/adaptive_difficulty.csv"

dta = pd.read_csv(dataset_path)

# defines the input features and targets
dta_input_features = [
    "player_health",
    "player_death_count",
    "time_survived",
    "player_noise",
    "player_energy",
    "average_enemy_dealt_time",
    "average_task_dealt_time",
    "player_success_rate",
    "average_hiding_time",
    "enemy_count",
    "enemy_spawn_rate",
    "average_chase_time",
    "average_detection_time",
    "area_danger_level",
    "level_task_progress",
    "task_spawn_timer"
]

dta_target = "recommended_difficulty"

# calculate the pearson correlation for input features with target
dta_correlation_matrix = dta[dta_input_features + [dta_target]].corr(method="pearson")

# correlation with target
dta_target_correlation = dta_correlation_matrix[dta_target].drop(dta_target)
dta_target_correlation = dta_target_correlation.sort_values(ascending=False)

print("\nFeature Correlation with reccomended difficulty:")
print(dta_target_correlation)

# plots the correlation heatmap
plt.figure(figsize=(18,14))

sns.heatmap(dta_correlation_matrix, annot=True, annot_kws={"size": 17}, fmt=".2f", cmap="coolwarm", center=0, linewidth=0.5)
plt.title("Adaptive Difficulty - Pearson Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/project_A_SVR/adaptive_difficulty_correlation.png", dpi=300)


# calculate the multicollinearity for input features only
dta_feature_correlation_matrix = dta[dta_input_features].corr(method="pearson")
threshold = 0.8

high_correlations = []

for i in range(len(dta_input_features)):
    for j in range(i +1, len(dta_input_features)):
        correlation = dta_feature_correlation_matrix.iloc[i,j]

        if abs(correlation) >= threshold:
            high_correlations.append({
                "feature 1": dta_input_features[i],
                "feature 2": dta_input_features[j],
                "correlation": correlation
            })
print("\n Input Feature pairs with high correlation:")

if high_correlations:
    dta_high_correlations = pd.DataFrame(high_correlations)
    print(dta_high_correlations.to_string(index=False))
else:
    print("No pair of Input Features with high correlation found")

plt.figure(figsize=(18,14))

sns.heatmap(dta_feature_correlation_matrix, annot=True, annot_kws={"size": 17}, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Input Features - Pearson Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/project_A_SVR/adaptive_difficulty_feature_correlation.png", dpi=300)
