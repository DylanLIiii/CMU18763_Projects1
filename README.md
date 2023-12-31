# FiFA 2015-2022 Analyst

This is a project contains following parts.

- Detailed EDA.
- Healthy data processing and machine learning pipeline.
- With pyspark and postgreSQL.

> A Video to walkthrough the project: https://cmu.box.com/s/dsq1lc9wty1u5zkt63mfypu3hgup89k3
## How to use it

You can choose to use the jupyter notebook or the python file. For Task1&2 using jupyter notebook, for Task3 using jupyter notebook and python scripts.

- **Notebook**: contains one EDA, one Task1&2 jupyter, one Data preprocessing(This is a notebook for data preprocessing, you can choose to use it or not, because I have already intergrate it with spark pipeline) and on Task3 jupyter notebook for Google Cloud.
- **Code**: contains utils, models, train.py. You can task3.py to run the whole pipeline.
- **Data**: contains the data I used in this project.

### Create your conda or pip env 

 `conda create --name <env> --file requirments.txt`

You should create your env for project.

### Using Jupyter Notebook

Just run jupyter notebooks in Notebook folder using python kernel in your env.

### Using Python Scripts

<details>
<summary>Params in scripts</summary>
- Verbose Mode:
Use --verbose followed by 0 or 1 to turn off or on verbose mode (more detailed output).
Example: --verbose 1
- wandb Logging:
Use --is_wandb followed by 0 or 1 to disable or enable wandb logging.
Example: --is_wandb 1
- SparkML Usage:
Use --is_sparkml followed by 0 or 1 to specify whether to use SparkML.
Example: --is_sparkml 1
- PyTorch Usage:
Use --is_pytorch followed by 0 or 1 to specify whether to use PyTorch.
Example: --is_pytorch 1
- Inference Mode:
Use --is_infer followed by 0 or 1 to select between inference mode (1) or test mode (0). Test mode uses a smaller dataset and fewer epochs.
Example: --is_infer 1
- Output Path:
Use --output_path followed by the path where you want to save the output.
Example: --output_path /path/to/output
</details>


For Task3, you can use python scripts in your env.

`python Task3.py /path/to/input --verbose 1 --is_wandb 0 --is_sparkml 1 --is_pytorch 0 --is_infer 0 --output_path /path/to/output`

Here I recommand a quick test to run it, you can use 

`python Task3.py /path/to/inputdata --verbose 1 --is_wandb 0 --is_sparkml 1 --is_pytorch 1 --is_infer 0 --output_path /path/to/outputmodel` to have a quick test. Set `is_infer` to 1 to run full inference pipeline.

> When you set `is_infer` to 0, it will use small data to train and test. When you set `is_infer` to 1, it will use full data to train and test.

You can use `python task3.py -h` to see the help information of arguments. You must specify the input_path (Your data path like `/data`)

---

Note that in `Task3.py` and `Task1&2.ipynb`, I set the environment variables of pyspark. If you need to run it locally, **please comment the environment variable settings and use the IP address that your local pyspark instance can bind to.**

## Model Performance

In this project, I use pyspark for General Machine Learning Model, utiliz pytorch for Deep Learning Model.

#### Train and Test

- train = 0.8 * Whole_data

- test = 0.2 * whole_data

#### ML Models

- Linear Regression
  - Reason to choose: **Use linear regression as baseline performance for comparison to facilitate implementation. Another reason is that from visual analysis, it can be found that features and overall have a strong correlation.**
  - Train RMSE With lr 0.01: 1.66797
  - Val RMSE With lr 0.01: 1.65661

- Decision Tree
  - Reason to choose: **Tree models are always proven to be more effective on tabular data, especially tree models powered by gradient boosting algorithms, and I chose decision trees to exceed baseline performance.**
  - Train RMSE With lr 0.01: 1.835102
  - Val RMSE With lr 0.01: 1.8603

**Conclusion**: We use Linear Regression as the baseline without adjusting too many parameters for individual models. We can see that Linear Regression is dominant. This is in line with our intuition. When the linear correlation between features and target is high, linear regression can Get good results. And decision trees are always good at dealing with non-linear related features.


#### DL Models

- MLP: An ordinary multi-layer perceptron model, using Adam as the optimizer and MSE as the loss function, with a learning rate of 0.0003 and training 300 epochs.
  - Reason to choose: A simple yet effective MLP model is common as a baseline, which facilitates comparison of performance and is easy to implement.
- Results
  - Hidden size: 1024, 512, 256, 64. LR: 0.003, Epoch:100, NO LR Scheduler.
    - Train Loss: 0.6765
    - Val Loss: 0.5828
  - Hidden size: 1024, 512, 256, 64. LR: 0.01, Epoch:100, with LR Scheduler.
    - Train Loss: 0.8225
    - Val Loss: 0.7001
- MLP with residual link(1024, 512, 512, 256, 64, 32, 16): An MLP with residual link (Kaiming He). using Adam as the optimizer and MSE as the loss function, with a learning rate of 0.0003 and training 100 epochs. 
  - Reason to choose: **Residual connections give us the opportunity to create deeper networks. The depth of the network should lead to better fitting performance.**
- Results:
  - Hidden size: 1024, 512, 512, 256, 64, LR: 0.003, Epoch:60, NO LR Scheduler.
    - Train Loss: 0.5704
    - Val Loss: 0.5759
  - Hidden size: 1024, 512, 512, 256, 256, 64, 32, 16. (Deeper) LR: 0.01, Epoch:60, with LR Scheduler.
    - Train Loss: 1.129
    - Val Loss: 1.844

**Conclusion**:  We use MLP as the baseline. We can see that when the epoch is high, both converge, even without the lr scheduler. However, the residual MLP that increases the learning rate, deepens the network, and trains for fewer epochs is worse. This may be This is because deeper networks require smaller learning rates and longer training rounds. But in general, neural networks perform better.

**SEE DETAILED LOG**: https://api.wandb.ai/links/dylanli/xfplxl1r

## Data Describtion

- Data Report: In dataview.html. You can download it and open in a web browser.
- Data EDA: [FIFA2022 EDA](https://www.kaggle.com/code/dylanhedded/fifa2022-eda)

<details>
<summary>Column Explaination</summary>

- `sofifa_id`: This is an integer that represents the unique ID of a player in the SoFIFA database.
- `player_url`: This is a string that contains the URL of a player's profile.
- `short_name`: This is a string representing the short name of the player.
- `long_name`: This is a string representing the full name of the player.
- `player_positions`: This string represents the positions the player can play in.
- `overall`: This integer represents the overall performance rating of the player.
- `potential`: This integer represents the potential performance rating of the player.
- `value_eur`: This double represents the market value of the player in Euros.
- `wage_eur`: This double represents the wage of the player in Euros.
- `age`: This integer represents the age of the player.
- `dob`: This date field represents the date of birth of the player.
- `height_cm`: This integer represents the height of the player in centimeters.
- `weight_kg`: This integer represents the weight of the player in kilograms.
- `club_team_id`: This double likely represents the unique ID of the club team the player belongs to.
- `club_name`: This string represents the name of the club team the player belongs to.
- `league_name`: This string represents the name of the league the club team competes in.
- `league_level`: This double likely represents the level or tier of the league the club team competes in.
- `club_position`: This string represents the position the player plays in at their club team.
- `club_jersey_number`: This double represents the jersey number of the player at their club team.
- `club_loaned_from`: This string represents the club team the player is loaned from, if applicable.
- `club_joined`: This date field represents the date when the player joined the current club.
- `club_contract_valid_until`: This double likely represents the year until which the player's contract with the club is valid.
- `nationality_id`: This integer likely represents a unique identifier for the player's nationality.
- `nationality_name`: This string represents the nationality of the player.
- `nation_team_id`: This double likely represents the unique ID of the national team the player belongs to.
- `nation_position`: This string represents the position the player plays in at their national team.
- `nation_jersey_number`: This double represents the jersey number of the player at their national team.
- `preferred_foot`: This string indicates the player's preferred foot (either 'Left' or 'Right').
- `weak_foot`: This integer represents the player's skill level with their non-dominant foot.
- `skill_moves`: This integer represents the number of skill moves the player can perform.
- `international_reputation`: This integer represents the player's reputation on an international level.
- `work_rate`: This string represents the player's work rate, typically represented as a combination of their attacking and defensive work rates.
- `body_type`: This string describes the player's body type.
- `real_face`: This string indicates whether the player has a real face in the game or not.
- `release_clause_eur`: This double represents the player's release clause in Euros, if applicable.
- `player_tags`: This string contains any special tags associated with the player.
- `player_traits`: This string contains any special traits that the player has.
- `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`: These doubles represent the player's skill ratings in these areas.
- `attacking_crossing`, `attacking_finishing`, `attacking_heading_accuracy`, `attacking_short_passing`, `attacking_volleys`: These integers represent various attacking attributes of the player.
- `skill_dribbling`, `skill_curve`, `skill_fk_accuracy`, `skill_long_passing`, `skill_ball_control`: These integers represent various skill attributes of the player.
- `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_reactions`, `movement_balance`: These integers represent various movement attributes of the player.
- `power_shot_power`, `power_jumping`, `power_stamina`, `power_strength`, `power_long_shots`: These integers represent various power attributes of the player.
- `mentality_aggression`, `mentality_interceptions`, `mentality_positioning`, `mentality_vision`, `mentality_penalties`, `mentality_composure`: These integers represent various mentality attributes of the player.
- `defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle`: These integers represent various defending attributes of the player- `goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_kicking`, `goalkeeping_positioning`, `goalkeeping_reflexes`: These integers represent various goalkeeping attributes of the player.
- `goalkeeping_speed`: This double represents the speed attribute of the player in goalkeeping.
- `ls`, `st`, `rs`, `lw`, `lf`, `cf`, `rf`, `rw`, `lam`, `cam`, `ram`, `lm`, `lcm`, `cm`, `rcm`, `rm`, `lwb`, `ldm`, `cdm`, `rdm`, `rwb`, `lb`, `lcb`, `cb`, `rcb`, `rb`, `gk`: These strings represent the player's skills ratings in different positions on the pitch.
- `player_face_url`: This string represents the URL of the player's face image.
- `club_logo_url`: This string represents the URL of the club's logo.
- `club_flag_url`: This string represents the URL of the club's flag.
- `nation_logo_url`: This string represents the URL of the nation's logo.
- `nation_flag_url`: This string represents the URL of the nation's flag.
- `year`: This integer represents the year of the data.
- `id`: This long integer likely represents a unique identifier for each row or record in the dataset.

</details>

<details>
<summary>some example constrains in SQL Database</summary>

- `_c0 INT PRIMARY KEY`: Defines `_c0` as an integer field that serves as the primary key.
- `sofifa_id INT NOT NULL`: Defines `sofifa_id` as an integer field that cannot be null.
- `player_url VARCHAR(255) NOT NULL`: Defines `player_url` as a string of up to 255 characters that cannot be null.
- `short_name VARCHAR(50) NOT NULL`: Defines `short_name` as a string of up to 50 characters that cannot be null.
- `long_name VARCHAR(100) NOT NULL`: Defines `long_name` as a string of up to 100 characters that cannot be null.
- `player_positions VARCHAR(50)`: Defines `player_positions` as a string of up to 50 characters.
- `overall INT NOT NULL`: Defines `overall` as an integer field that cannot be null.
- `potential INT NOT NULL`: Defines `potential` as an integer field that cannot be null.
- `value_eur DOUBLE NOT NULL`: Defines `value_eur` as a double precision number that cannot be null.
- `wage_eur DOUBLE NOT NULL`: Defines `wage_eur` as a double precision number that cannot be null.
- `age INT NOT NULL`: Defines `age` as an integer field that cannot be null.
- `dob DATE NOT NULL`: Defines `dob` as a date field that cannot be null.
- `height_cm INT NOT NULL`: Defines `height_cm` as an integer field that cannot be null.
- `weight_kg INT NOT NULL`: Defines `weight_kg` as an integer field that cannot be null.
- `club_team_id INT`: Defines `club_team_id` as an integer field.
- `club_name VARCHAR(50)`: Defines `club_name` as a string of up to 50 characters.
- `league_name VARCHAR(50)`: Defines `league_name` as a string of up to 50 characters.
- `league_level INT`: Defines `league_level` as an integer field.
- `club_position VARCHAR(50)`: Defines `club_position` as a string of up to 50 characters.
- `club_jersey_number INT`: Defines `club_jersey_number` as an integer field.
- `club_loaned_from VARCHAR(50)`: Defines `club_loaned_from` as a string of up to 50 characters.
- `club_joined DATE`: Defines `club_joined` as a date field.
- `club_contract_valid_until INT`: Defines `club_contract_valid_until` as an integer field.
- `nationality_id INT`: Defines `nationality_id` as an integer field.
- `nationality_name VARCHAR(50)`: Defines `nationality_name` as a string of up to 50 characters.
- `nation_team_id INT`: Defines `nation_team_id` as an integer field.
- `nation_position VARCHAR(50)`: Defines `nation_position` as a string of up to 50 characters.
- `nation_jersey_number INT`: Defines `nation_jersey_number` as an integer field.

</details>
