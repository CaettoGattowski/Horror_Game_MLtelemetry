import pandas as pd

dataset_path = "datasets/adaptive_difficulty.csv"

dtv = pd.read_csv(dataset_path)

print("Dataset Validation")
print("=" * 40)

# checks shape
print("\n Dataset Shape:")
print(dtv.shape)

# checks missing values
print("\n Dataset's Missing Values:")
dtv_missing_values = dtv.isnull().sum()
print(dtv_missing_values)
print("\n Total Missing Values are:", dtv_missing_values.sum())

# checks duplicated values
print("\n Dataset's Duplicated Values:")
dtv_duplicated_values = dtv.duplicated().sum()
print(dtv_duplicated_values)
print("\n Total Duplicated Values are:", dtv_duplicated_values.sum())

# checks all the features and their ranges
dtv_feature_range = {
    "player_health": (0, 100),
    "player_death_count": (0, 10),
    "time_survived": (0, 300),
    "player_noise": (0, 3),
    "player_energy": (0, 100),
    "average_enemy_dealt_time": (0, 20),
    "average_task_dealt_time": (0, 20),
    "player_success_rate": (0, 100),
    "average_hiding_time": (0, 30),
    "enemy_count": (0, 10),
    "enemy_spawn_rate": (0, 30),
    "average_chase_time": (0, 30),
    "average_detection_time": (0, 20),
    "area_danger_level": (0, 1),
    "level_task_progress": (0, 100),
    "task_spawn_timer": (0, 60),
    "recommended_difficulty": (0, 100)
}

print("\nFeature range validation:")

for feature, (minimum, maximum) in dtv_feature_range.items():

    invalid_values = dtv[
        (dtv[feature] < minimum) |
        (dtv[feature] > maximum)
    ]

    if invalid_values.empty:
        print(f"{feature}: OK")
    else:
        print(f"{feature}: {len(invalid_values)} invalid values")

# 6. Check classification labels
dtv_valid_categories = ["Easy", "Normal", "Hard", "Extreme"]

dtv_invalid_categories = dtv[~dtv["difficulty_category"].isin(dtv_valid_categories)]

print("\nInvalid difficulty categories:")
print(len(dtv_invalid_categories))

# 7. Class distribution
print("\nDifficulty category distribution:")

dtv_class_counts = dtv["difficulty_category"].value_counts()

dtv_class_percentages = dtv["difficulty_category"].value_counts(
    normalize=True
) * 100

dtv_distribution = pd.DataFrame({
    "Count": dtv_class_counts,
    "Percentage": dtv_class_percentages
})

print(dtv_distribution)
print("\nValidation completed.")
