# CS8803 Project Midterm Deliverable - A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

## Set Up Instructions
Create a conda environment using "env.yml"

### Training
All training scripts are located in `train/`
1. `dagger.py`: train baseline DAgger on LunarLanderV2
2. `safe_dagger.py`: train DAgger on LunarLanderV2 with safety aware dataset aggregation
3. `dagger_po.py`: train DAgger on partially observable LunarLanderV2
    - use hyperparameter constants at top of file to configure training
4. `dagger_po_lstm.py`: train DAgger on partially observable LunarLanderV2 using LSTM as recurrent architecture
    - use hyperparamter constants at top of file to configure training
5. `safe_dagger_po.py`: train DAgger on partially observable LunarLanderV2 using safety aware dataset aggregation
    - use hyperparameter constants at top of file to configure training

Run all training scripts from the `train/` folder!

### Results
All generated results can be found in `results/` under their relevant experiment name
1. `exp-robustness/` contains the results for MLPs vs GRUs in partially observable LunarLanderV2 across different levels of observation dropout
2. `exp-lstm` contains the results for GRUs vs MLPs in partially observable LunarLanderV2
3. `exp-safe-dagger/` contains the results for comparing DAgger against DAgger with safety aware dataset aggregation