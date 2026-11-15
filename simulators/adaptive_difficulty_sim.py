import numpy as np
import pandas as pd

NUM_SAMPLES = 30000
RANDOM_SEED = 42
OUTPUT_FILE = "datasets/adaptive_difficulty.csv"

# DATASET SIM #

def generate_dataset( num_samples=NUM_SAMPLES, seed=RANDOM_SEED
) -> pd.DataFrame:
    # RANDOM NUMBER GENERATOR
    rng = np.random.default_rng(seed)


    # HIDDEN GAME VARIABLES
    # these can play as soft constaints during the simulation and the generation of our data=
    # hypothetical player skill varrying from 0 to 1 where 0 = begginner, 1 = expert skill
    player_skill = rng.uniform(0, 1, num_samples)
    # hypothetical game pressure varrying from 0 to 1 where 0 = no pressure, 1 = high pressure
    game_pressure = rng.uniform(0, 1, num_samples)

    # PLAYER FEATURES
    # we generate the random sample variables from a NORMAL (gaussian) distribution and CLIPPING them to limit the values in the array but also where need be we use ROUNDING to evenly round the given number of decimals
    # time values are in seconds and we limit them to a maximum of 300 seconds (5 minutes) for the level

    # number of player's health points
    player_health = np.clip(np.round(100 - 35 * game_pressure + 20 * player_skill + rng.normal(0, 10, num_samples)), 0, 100).astype(int)
    # number of deaths / redos of the level
    player_death_count = np.clip(np.round(6 - 6 * player_skill + 2 * game_pressure + rng.normal(0, 1, num_samples)), 0, 10).astype(int)
    # time that the player survived in the level without losing / dying in seconds
    time_survived = np.clip(300 - 300 * game_pressure + 100 * player_skill + rng.normal(0, 50, num_samples), 0, 300).astype(int)
    # number of accumulated noise the player made with actions like jumping, running etc.
    player_noise = np.clip(np.round(3 - 3 * player_skill + 2 * game_pressure + rng.normal(0, 1, num_samples)), 0, 3).astype(int)
    # number of energy / stamina the player has
    player_energy = np.clip(np.round(100 - 20 * game_pressure + 10 * player_skill + rng.normal(0, 10, num_samples)), 0, 100).astype(int)
    # average reaction time it takes for a player to deal with a spawned enemy
    average_reaction_enemy_time = np.clip(20 - 5 * player_skill + 10 * game_pressure + rng.normal(0, 1, num_samples), 0, 20)
    # average reaction time it takes for a player to deal with a task
    average_reaction_task_time = np.clip(10 - 5 * player_skill + 10 * game_pressure + rng.normal(0, 1, num_samples), 0, 20)
    # number of continuous successful completions of a level
    player_success_rate = np.clip(np.round(3 + 3 * player_skill - 4 * game_pressure + rng.normal(0, 1, num_samples)), 0, 5).astype(int)
    # time the player spent hiding from enemies during the level
    average_hiding_time = np.clip(15 - 10 * player_skill + 20 * game_pressure + rng.normal(0, 5, num_samples), 0, 30)
    # average time the player spent running around the level
    average_running_time = np.clip(60 - 20 * player_skill + 30 * game_pressure + rng.normal(0, 10, num_samples), 0, 120)
    # average time the player spent walking around the level
    average_walking_time = np.clip(45 - 15 * player_skill + 10 * game_pressure + rng.normal(0, 5, num_samples), 0, 90)
    # average time the player spent crouching during the level
    average_crouching_time = np.clip(60 - 30 * player_skill + 20 * game_pressure + rng.normal(0, 5, num_samples), 0, 60)
    # average time the player spent jumping + on_air during the level
    average_jumping_time = np.clip(30 - 10 * player_skill + 10 * game_pressure + rng.normal(0, 1, num_samples), 0, 30)

    # ENEMY FEATURES
    # number of spawned enemies at once during a delta time  of the game
    enemy_count = np.clip(np.round(4 + 2 * game_pressure - 4 * player_skill + rng.normal(0, 1, num_samples)), 0, 8).astype(int)
    # rate with which enemies spawn during the level sequence
    enemy_spawn_rate = np.clip(15 - 10 * game_pressure + rng.normal(0, 3, num_samples), 0, 30)
    # average time it takes an enemy chasing the player
    average_chase_time = np.clip(20 - 10 * player_skill + 20 * game_pressure + rng.normal(0, 5, num_samples), 0, 30)
    # average time it takes for an enemy to notice the player after spawning
    average_aggression_time = np.clip(10 - 5 * player_skill + 10 * game_pressure + rng.normal(0, 2, num_samples), 0, 20)

    # ENVIROMENTAL AND PROGRESS FEATURES
    # define parts of the room as dangerous and safe areas, e.g under the bed is safer than the middle of the room
    area_danger_level = np.clip(rng.uniform(0, 1, num_samples), 0, 1)
    # progress percentage of the tasks
    level_task_progress = np.clip(np.round(50 - 20 * game_pressure + 30 * player_skill + rng.normal(0, 1, num_samples)), 0, 100).astype(int)
    # time it takes for a task to spawn in the level
    task_spawn_timer = np.clip(30 - 10 * game_pressure + rng.normal(0, 5, num_samples), 0, 60)

    # TARGET: RECOMMENDED DIFFICULTY
    # skill score estimation based on observed player performance and game pressure, we can use this score to recommend a difficulty level for the player
    skill_score = (0.35 * player_success_rate + 0.25 * (player_success_rate /10) + 0.20 * (average_reaction_task_time / 10) + 0.15 * (average_reaction_enemy_time / 10) + 0.10 * area_danger_level)
    # pressure score estimation based on observable game features
    pressure_score = (0.35 * (player_death_count / 10) + 0.30 * (enemy_count / 10) + 0.25 * (enemy_spawn_rate / 10) + 0.20 * (task_spawn_timer / 10) + 0.15 * (enemy_spawn_rate / 10) + 0.10 * area_danger_level)
    # recommended difficulty score based on the skill and pressure scores, we can use this score to recommend a difficulty level for the player
    recommended_difficulty = np.clip(100 * (0.95 * skill_score - 0.60 * pressure_score) + rng.normal(0, 3, num_samples), 0, 100)

    # TARGET: DIFFICULTY CATEGORY
    difficulty_category = pd.cut(recommended_difficulty, bins=[0, 25, 50, 75, 100], labels=['Easy', 'Normal', 'Hard', 'Extreme'])

    # CREATE DATAFRAME

    dataset = pd.DataFrame({
        'player_health': player_health,
        'player_death_count': player_death_count,
        'time_survived': time_survived,
        'player_noise': player_noise,
        'player_energy': player_energy,
        'average_reaction_enemy_time': average_reaction_enemy_time,
        'average_reaction_task_time': average_reaction_task_time,
        'player_success_rate': player_success_rate,
        'average_hiding_time': average_hiding_time,
        'average_running_time': average_running_time,
        'average_walking_time': average_walking_time,
        'average_crouching_time': average_crouching_time,
        'average_jumping_time': average_jumping_time,
        'enemy_count': enemy_count,
        'enemy_spawn_rate': enemy_spawn_rate,
        'average_chase_time': average_chase_time,
        'average_aggression_time': average_aggression_time,
        'area_danger_level': area_danger_level,
        'level_task_progress': level_task_progress,
        'task_spawn_timer': task_spawn_timer,
        'recommended_difficulty': recommended_difficulty,
        'pressure_score': pressure_score,
        'difficulty_category': difficulty_category
    })

    return dataset

# MAIN #

if __name__ == "__main__":
    dataset = generate_dataset()

    # save the dataset to a csv file #
    dataset.to_csv(OUTPUT_FILE, index=False)
    print("Dataset generated successfully")
    # prints legth of the dataset to verify the number of samples generated
    print(f"Number of samples: {len(dataset)}")
    # prints the number of columns in the dataset to verify the number of features generated
    print(f"Number of columns: {len(dataset.columns)}")

    print("\nFirst 5 records:")
    print(dataset.head())

    print("\nDifficulty catergory distribution:")
    print(dataset['difficulty_category'].value_counts())

    print("\nDataset Info:")
    print(dataset.describe())

    print(f"\nDataset saved to {OUTPUT_FILE}")
