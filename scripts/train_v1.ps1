param(
    [int]$Timesteps = 100000,
    [ValidateSet("random", "starter", "greedy")]
    [string]$Opponent = "starter",
    [string]$SavePath = "models/orbit_wars_maskableppo_v1.zip",
    [string]$TensorboardLog = "runs",
    [int]$Seed = 42
)

python -m orbit_wars_rl.train --timesteps $Timesteps --opponent $Opponent --save-path $SavePath --tensorboard-log $TensorboardLog --seed $Seed
