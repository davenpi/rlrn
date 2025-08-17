Implement REINFORCE (simple policy gradient). Outputs are:

```zsh
(rlrn) rlrn % uv run python reinforce/alg.py 
Update 1: Avg return = 19.9, Min = 9.0, Max = 56.0
Update 10: Avg return = 23.7, Min = 9.0, Max = 59.0
Update 20: Avg return = 32.7, Min = 11.0, Max = 108.0
Update 30: Avg return = 36.6, Min = 11.0, Max = 131.0
Update 40: Avg return = 45.0, Min = 12.0, Max = 172.0
Update 50: Avg return = 50.9, Min = 12.0, Max = 145.0
Update 60: Avg return = 70.4, Min = 16.0, Max = 228.0
Update 70: Avg return = 87.8, Min = 24.0, Max = 327.0
Update 80: Avg return = 109.6, Min = 17.0, Max = 287.0
Update 90: Avg return = 137.4, Min = 26.0, Max = 387.0
Update 100: Avg return = 176.7, Min = 22.0, Max = 482.0
Update 110: Avg return = 236.1, Min = 34.0, Max = 500.0
Update 120: Avg return = 267.3, Min = 44.0, Max = 500.0
Update 130: Avg return = 337.4, Min = 92.0, Max = 500.0
Update 140: Avg return = 351.4, Min = 113.0, Max = 500.0
Update 150: Avg return = 350.5, Min = 113.0, Max = 500.0
Update 160: Avg return = 434.0, Min = 125.0, Max = 500.0
Update 170: Avg return = 453.5, Min = 88.0, Max = 500.0
Update 180: Avg return = 435.8, Min = 155.0, Max = 500.0
Update 190: Avg return = 473.5, Min = 205.0, Max = 500.0
Update 200: Avg return = 465.4, Min = 208.0, Max = 500.0
```

Demonstrates REINFORCE is sample inefficient. We need to:
- Generate many entire trajectories to get a good estimate of the gradient
- We can't learn until trajectories are complete
- Training takes longer the better we get (longer episodes means longer time until model updates)