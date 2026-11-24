import pandas as pd

# loads the dataset
dataset_path = "datasets/threat_encounter.csv"

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
    "enemy_patrol_angle": (0, 360),
    "enemy_speed": (0, 3),
    "enemy_acceleration": (0, 6),
    "enemy_detection_radius": (0, 20),
    "enemy_fov": (90, 360),
    "enemy_sound_sensitivity": (30, 60),
    "obstacle_density": (0, 1),
    "lighting_intensity": (0, 100),
    "area_hiding_spot_availability": (0, 1),
    "hiding_spot_number": (0, 5),
    "task_number": (0, 20),
    "time_to_next_threat_event": (0, 120)
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
    
# checks classification labales
dtv_valid_categories = {
    "Patrol",
    "Investigation",
    "Chase",
    "Attack"
}

dtv_found_categories = set(dtv["encounter_category"].unique())

dtv_invalid_categories = dtv_found_categories - dtv_valid_categories

print("Found categories:", dtv_found_categories)
print("Invalid categories:", dtv_invalid_categories)

if len(dtv_invalid_categories) == 0:
    print("PASS: All encounter categories are valid.")
else:
    print("FAIL: Invalid encounter categories detected.")

# Category distribution
print("\n6. Encounter category distribution:")
print(dtv["encounter_category"].value_counts())

print("\nValidation completed")