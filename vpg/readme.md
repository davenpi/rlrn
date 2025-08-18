Using advantage is `reward-to-go - V(s_t)`

Learning:
```zsh
(rlrn) rlrn % uv run python vpg/alg.py
Update 0: Avg return = 18.0, Min return = 8.0, Max return = 50.0, Value loss = 207.357
Update 10: Avg return = 24.4, Min return = 9.0, Max return = 101.0, Value loss = 559.184
Update 20: Avg return = 27.7, Min return = 10.0, Max return = 100.0, Value loss = 602.975
Update 30: Avg return = 35.8, Min return = 13.0, Max return = 99.0, Value loss = 775.133
Update 40: Avg return = 47.6, Min return = 12.0, Max return = 292.0, Value loss = 3084.583
Update 50: Avg return = 66.0, Min return = 14.0, Max return = 244.0, Value loss = 3804.542
Update 60: Avg return = 72.9, Min return = 16.0, Max return = 237.0, Value loss = 3738.025
Update 70: Avg return = 117.0, Min return = 16.0, Max return = 376.0, Value loss = 10019.382
Update 80: Avg return = 166.7, Min return = 31.0, Max return = 500.0, Value loss = 19073.320
Update 90: Avg return = 229.4, Min return = 29.0, Max return = 500.0, Value loss = 31972.430
Update 100: Avg return = 290.4, Min return = 88.0, Max return = 500.0, Value loss = 42921.000
Update 110: Avg return = 324.4, Min return = 113.0, Max return = 500.0, Value loss = 47809.973
Update 120: Avg return = 388.1, Min return = 93.0, Max return = 500.0, Value loss = 62531.031
Update 130: Avg return = 427.8, Min return = 90.0, Max return = 500.0, Value loss = 68742.578
Update 140: Avg return = 466.5, Min return = 250.0, Max return = 500.0, Value loss = 75313.984
Update 150: Avg return = 476.9, Min return = 221.0, Max return = 500.0, Value loss = 77650.328
Update 160: Avg return = 483.7, Min return = 300.0, Max return = 500.0, Value loss = 77406.766
Update 170: Avg return = 494.3, Min return = 182.0, Max return = 500.0, Value loss = 79925.086
Update 180: Avg return = 479.0, Min return = 84.0, Max return = 500.0, Value loss = 77006.742
Update 190: Avg return = 480.7, Min return = 127.0, Max return = 500.0, Value loss = 78019.742
Update 199: Avg return = 493.4, Min return = 286.0, Max return = 500.0, Value loss = 78756.328
```

Value function (critic) does terribly! Hard because the underlying distribution changes as the
policy updates. Still useful for reducing variance over having raw `reward-to-go` weight.