Reimplementing basic RL algos in Python

By Aug 29
- [x] REINFORCE
- [ ] VPG
- [ ] PPO
- [ ] DQN
- [ ] SAC

## Notes for self/others

- Spinning Up derivation of baseline formula is a bit confusing to me. Easier to start with regular policy gradient formula,
use linearity of expectation value, then use "law of total expectation" to split into nested expectations (one over states and one over actions).
The inner expectation over actions is zero.