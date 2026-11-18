import numpy as np
import pandas as pd
from pathlib import Path

NUM_SAMPLES = 3000
RANDOM_SEED = 42
MAX_THREAT_TIME = 120
OUTPUT_FILE = "datasets/threat_encounter.csv"

# DATASET SIM #
def generate_dataset( num_samples=NUM_SAMPLES, seed=RANDOM_SEED, max_threat_time=MAX_THREAT_TIME
) -> pd.DataFrame:
    # RANDOM NUMBER GENERATOR
    rng = np.random.default_rng(seed)

    # DEFINE ENEMY PROFILES
    # we created fixed behavioural parameters for each enemy type
    enemy_profiles_config = {
        "Stalker": {
            "enemy_speed": 1.0,
            "enemy_acceleration": 2,
            "enemy_detection_radius": 12,
            "enemy_fov": 90,
            "enemy_sound_sensitivity": 40,
            "enemy_aggression": 30
        },

        "Hunter": {
            "enemy_speed": 1.5,
            "enemy_acceleration": 4,
            "enemy_detection_radius": 9,
            "enemy_fov": 120,
            "enemy_sound_sensitivity": 50,
            "enemy_aggression": 50
        },

        "Pursuer": {
            "enemy_speed": 3.0,
            "enemy_acceleration": 6,
            "enemy_detection_radius": 6,
            "enemy_fov": 180,
            "enemy_sound_sensitivity": 60,
            "enemy_aggression": 70
        },

        "Ambusher": {
            "enemy_speed": 0.5,
            "enemy_acceleration": 2,
            "enemy_detection_radius": 10,
            "enemy_fov": 360,
            "enemy_sound_sensitivity": 40,
            "enemy_aggression": 100
        }
    }

    profile_names = list(enemy_profiles_config.keys())
    # 0.35 for stalker, 0.30 for hunter, 0.20 for pursuer, 0.15 for ambusher
    profile_probabilities = [0.35, 0.30, 0.20, 0.15]

    enemy_profiles = rng.choice(profile_names, size=num_samples, p=profile_probabilities)
    profile_counts = pd.Series(enemy_profiles).value_counts()

    # GENERATE ENVIROMENTAL FEATURES
    # angle the enemy looks around during patrol
    enemy_patrol_angle = rng.uniform(0, 360, num_samples)
    # defines if there is obstacle density in the specific area with yes 1 or no 0
    obstacle_density = rng.uniform(0, 1, num_samples)
    # defines how long is the corridor
    corridor_width = rng.uniform(1, 10, num_samples)
    # defines how far away is something visible, like fog of war / darkness or not
    visibility_distance = rng.uniform(1, 50, num_samples)
    # defines the lighting in that area in terms of luminosity
    lighting_intensity = rng.uniform(0, 100, num_samples)
    # defines if there are available hiding spots in that specific area
    area_hiding_spot_availability = rng.uniform(0, 1, num_samples)
    # checks how many numbers of hiding spots exist in that specific area
    hiding_spot_number = rng.integers(0, 6, num_samples)
    # checks the distance of all hiding spots in that area
    hiding_spot_distance = rng.uniform(0, 50, num_samples)
    # defines random noise + player noise that exists in the enviroment
    enviromental_noise = rng.uniform(0, 100, num_samples)
    # checks the number of tasks
    task_number = task_number = rng.integers(0, 21, num_samples)
    
    # FIXED PARAMETERS
    enemy_speed = np.array([enemy_profiles_config[p]["enemy_speed"] for p in enemy_profiles])
    enemy_acceleration = np.array([enemy_profiles_config[p]["enemy_acceleration"] for p in enemy_profiles])
    enemy_detection_radius = np.array([enemy_profiles_config[p]["enemy_detection_radius"] for p in enemy_profiles])
    enemy_fov = np.array([enemy_profiles_config[p]["enemy_fov"] for p in enemy_profiles])
    enemy_sound_sensitivity = np.array([enemy_profiles_config[p]["enemy_sound_sensitivity"] for p in enemy_profiles])
    enemy_aggression = np.array([enemy_profiles_config[p]["enemy_aggression"] for p in enemy_profiles])

    # GENERATE INTERNAL SIMULATION VARIABLES
    # these variables are not saved as dataset input features
    # distance between player and enemy
    player_enemy_distance = rng.uniform(1, 50, num_samples)
    # the player's direction is defined as 0 degress in the coordinate system
    player_bearing = 0.0
    # calculates the smallest angular difference between the enemy's patrol direction and the player's direcrion
    relative_enemy_angle = np.abs((player_bearing - enemy_patrol_angle + 180) % 360-180)

    # GENERATE time_to_next_threat_event for regression
    # this is a continuous target measured in seconds it is calculated independantly of encounter_category
    time_to_next_threat_event = []

    for i in range(num_samples):
        
        speed = enemy_speed[i]
        acceleration = enemy_acceleration[i]

        distance = player_enemy_distance[i]
        relative_angle = relative_enemy_angle[i]

        obstacles = obstacle_density[i]
        corridor = corridor_width[i]

        visibility = visibility_distance[i]
        lighting = lighting_intensity[i]

        hiding_availability = area_hiding_spot_availability[i]
        hiding_number = hiding_spot_number[i]
        hiding_distance = hiding_spot_distance[i]

        noise = enviromental_noise[i]
        tasks = task_number[i]

        sound_sensitivity = enemy_sound_sensitivity[i]

        # base travel time | a faster enemy reaches the player sooner.
        movement_time = distance / max(speed, 0.1)

        # acceleration reduces the effective approach time
        acceleration_factor = 1 / (1 + acceleration / 10)

        # obstacles and narrow corridors affect movement
        obstacle_factor = (1 + obstacles * 0.8 + max(0, 5 - corridor) / 10)

        # a player outside the enemy's patrol direction is harder to approach and detect.
        orientation_factor = (1 + relative_angle / 180 * 0.3)

        # aalculate how effective the available hidingspots are.
        # nearby and numerous hiding spots providebetter opportunities to avoid a threat.
        hiding_effectiveness = (hiding_availability* (hiding_number / 5)* (1 - hiding_distance / 50))
        hiding_effectiveness = np.clip(hiding_effectiveness,0,1)
        hiding_factor = (1 + 0.35 * hiding_effectiveness)

        # better visibility and stronger lighting make it easier for the enemy to locate the player
        visibility_factor = 1 / (1 + (visibility / 50) * (lighting / 100))

        # sound can make the player easier to locate
        noise_factor = 1 / (1 + (noise / 100) * enemy_sound_sensitivity[i] / 100)

        # task distance can reduce or increase the threat
        task_factor = 1 / ( 1 + (tasks / 20) * 0.2)

        threat_time = (movement_time
            * acceleration_factor
            * obstacle_factor
            * orientation_factor
            * hiding_factor
            * visibility_factor
            * noise_factor
            * task_factor)

        # small simulation variability
        threat_time += rng.normal(0, 2)
        threat_time = np.clip(threat_time, 0, max_threat_time)

        time_to_next_threat_event.append(threat_time)
    time_to_next_threat_event = np.array(time_to_next_threat_event)

    # GENERATE encounter_category for classification 
    # Patrol(no immediate visual or sound detection) | Investigation(enemy detects a sound stimuli)
    # Chase(enemy visually detects the player) | Attack(enemy detects the player at close range and has sufficient high aggression)

    encounter_category = []

    for i in range(num_samples):
        profile = enemy_profiles[i]

        distance = player_enemy_distance[i]
        relative_angle = relative_enemy_angle[i]

        detection_radius = enemy_detection_radius[i]
        field_of_view = enemy_fov[i]

        sound_sensitivity = enemy_sound_sensitivity[i]
        aggression = enemy_aggression[i]

        # obstacles reduce effective visibility
        effective_visibility = (visibility_distance[i] * (1 - 0.5 * obstacle_density[i]))

        # visual detection conditions
        visual_detection = (distance <= detection_radius and distance <= effective_visibility and relative_angle <= field_of_view / 2 and lighting_intensity[i] > 20)

        # sound detection conditions
        sound_detection = (enviromental_noise[i] * sound_sensitivity / 100 > 25)

        # ambushers can attack at close range even withhout the usual visual detection condition
        ambush_attack = (profile == "Ambusher" and distance <= detection_radius and aggression >= 90) 

        if ( visual_detection and distance < 5 and aggression >= 60) or ambush_attack:
            category = "Attack"
        elif visual_detection:
            category = "Chase"
        elif sound_detection:
            category = "Investigation"
        else:
            category = "Patrol"

        encounter_category.append(category)

    # CREATE DATAFRAME
    dataset = pd.DataFrame({
        'enemy_patrol_angle': enemy_patrol_angle,
        'enemy_speed': enemy_speed,
        'enemy_acceleration': enemy_acceleration,
        'enemy_detection_radius': enemy_detection_radius,
        'enemy_fov': enemy_fov,
        'enemy_sound_sensitivity': enemy_sound_sensitivity,

        'obstance_density': obstacle_density,
        'corridor_width': corridor_width,
        'visibility_distance': visibility_distance,
        'lighting_intensity': lighting_intensity,
        'area_hiding_spot_availability': area_hiding_spot_availability,
        'hiding_spot_number': hiding_spot_number,
        'hiding_spot_distance': hiding_spot_distance,
        'enviromental_noise': enviromental_noise,
        'task_number': task_number,

        'time_to_next_threat_event': time_to_next_threat_event,
        'encounter_category': encounter_category
    })
    return dataset

# MAIN #
if __name__ == "__main__":
    dataset = generate_dataset()

    # save the dataset to a csv file #
    dataset.to_csv(OUTPUT_FILE, index=False)
    print("\nDataset generated successfully!")

    print("\nDataset shape:")
    print(dataset.shape)

    print("\nFirst five records:")
    print(dataset.head())

    print("\nEncounter category distribution:")
    print(dataset["encounter_category"].value_counts())

    print("\nRegression target statistics:")
    print(dataset["time_to_next_threat_event"].describe())

    print("\nMissing values:")
    print(dataset.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(dataset.duplicated().sum())

    print("\nTarget values equal to zero:")
    print((dataset["time_to_next_threat_event"] == 0).sum())

    print("\nTarget values equal to maximum:")
    print((dataset["time_to_next_threat_event"] == MAX_THREAT_TIME).sum())

    print("\nUnique hiding spot numbers:")
    print((sorted(dataset["hiding_spot_number"].unique())))

    print("\nUnique task numbers:")
    print(sorted(dataset["task_number"].unique()))