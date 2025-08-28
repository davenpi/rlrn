Reimplementing basic RL algos in Python

- [x] REINFORCE
- [x] VPG
- [ ] PPO variants
- [ ] Something off policy and more sample efficient

## Notes for self/others

- Spinning Up derivation of baseline formula is a bit confusing to me. Easier to start with regular policy gradient formula,
use linearity of expectation value, then use "law of total expectation" to split into nested expectations (one over states and one over actions).
The inner expectation over actions is zero. Can use the same ideas to get the "reward to go" formula (expand total return into a sum and note that R_k is indendpent of A_t for k < t)