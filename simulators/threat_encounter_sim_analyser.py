import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loads the dataset
dataset_path = "datasets/threat_encounter.csv"

dta = pd.read_csv(dataset_path)

# defines the input features and targets
dta_input_features = [
    "enemy_patrol_angle",
    "enemy_speed",
    "enemy_acceleration",
    "enemy_detection_radius",
    "enemy_fov",
    "enemy_sound_sensitivity",
    "obstacle_density",
    "corridor_width",
    "visibility_distance",
    "lighting_intensity",
    "area_hiding_spot_availability",
    "hiding_spot_number",
    "hiding_spot_distance",
    "environmental_noise",
    "task_number"
]

dta_target = "time_to_next_threat_event"

# calculates the pearson correlation for input features with target
dta_correlation_matrix = dta[dta_input_features + [dta_target]].corr(method="pearson")
# correlation with target
dta_target_correlation = dta_correlation_matrix[dta_target].drop(dta_target)
dta_target_correlation = dta_target_correlation.sort_values(ascending=False)

print("\nFeature Correlation with")
print(dta_target_correlation)

# plots the correlation heatmap
plt.figure(figsize=(18, 14))

sns.heatmap(dta_correlation_matrix, annot=True, annot_kws={"size": 17}, fmt=".2f", cmap="coolwarm", center=0, linewidth=0.5)
plt.title("Threat Encounter - Pearson Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/simulators/adaptive_difficulty_correlation.png", dpi=300)

# cacluates the multicollinearity for input features only
dta_feature_correlation_matrix = dta[dta_input_features].corr(method="pearson")
threshold = 0.8

high_correlations = []

for i in range(len(dta_input_features)):
    for j in range(i + 1, len(dta_input_features)):
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

plt.figure(figsize=(18, 14))

sns.heatmap(dta_feature_correlation_matrix, annot=True, annot_kws={"size": 17}, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Threat Encounter - Pearson Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/simulators/threat_encounter_feature_correlation.png", dpi=300)
