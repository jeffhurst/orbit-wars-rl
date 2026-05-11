param(
    [string]$ModelPath = "models/orbit_wars_maskableppo_v1.zip",
    [int]$Games = 10,
    [ValidateSet("random", "starter", "greedy")]
    [string]$Opponent = "starter",
    [int]$Seed = 42
)

python -m orbit_wars_rl.evaluate --model-path $ModelPath --games $Games --opponent $Opponent --seed $Seed
