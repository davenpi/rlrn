Implement REINFORCE (simple policy gradient). Outputs are:

```zsh
Update 0: Avg return = 22.8, Min = 9.0, Max = 59.0
Update 10: Avg return = 33.5, Min = 10.0, Max = 103.0
Update 20: Avg return = 37.1, Min = 11.0, Max = 120.0
Update 30: Avg return = 46.1, Min = 12.0, Max = 145.0
Update 40: Avg return = 53.3, Min = 11.0, Max = 168.0
Update 50: Avg return = 61.4, Min = 17.0, Max = 166.0
Update 60: Avg return = 73.1, Min = 23.0, Max = 201.0
Update 70: Avg return = 88.6, Min = 13.0, Max = 221.0
Update 80: Avg return = 118.1, Min = 25.0, Max = 291.0
Update 90: Avg return = 155.9, Min = 25.0, Max = 332.0
Update 100: Avg return = 206.4, Min = 18.0, Max = 500.0
Update 110: Avg return = 274.4, Min = 22.0, Max = 500.0
Update 120: Avg return = 304.8, Min = 94.0, Max = 500.0
Update 130: Avg return = 360.6, Min = 99.0, Max = 500.0
Update 140: Avg return = 376.8, Min = 67.0, Max = 500.0
Update 150: Avg return = 397.3, Min = 116.0, Max = 500.0
...
```

Demonstrates REINFORCE is sample inefficient. We need to:
- Generate many entire trajectories to get a good estimate of the gradient
- We can't learn until trajectories are complete
- Training takes longer the better we get (longer episodes means longer time until model updates)