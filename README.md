# CS258-Final-Project

python app\dqn_runner.py --debug=1 --train=app\dqn_rsaenv_model_cap20_tuned.zip --use_tuning --link_capacity=20
python app\dqn_runner.py --debug=1 --eval=app\dqn_rsaenv_model_cap20_tuned.zip --link_capacity=20

python app\dqn_runner.py --debug=1 --train=app\dqn_rsaenv_model_cap10_tuned.zip --use_tuning --link_capacity=10
python app\dqn_runner.py --debug=1 --eval=app\dqn_rsaenv_model_cap10_tuned.zip --link_capacity=10

## How to Execute

## Environment

### State Representation & State Transitions

### Action Representation

The agent has 8 actions to take.
Actions are stored in the following data structure -> `list[dict[str:int,str:int,str:list[int]]]`
It is essentially a list of actions, and each action is a dictionary that contains the source node, the destination node, and a list edges needed to be traversed to go from the source to destination.

**ACTION 1**  
source node: 0  
destination node: 3  
path: [Edge(0, 1), Edge(1, 2), Edge(2, 3)]

**ACTION 2**  
source node: 0  
destination node: 3  
path: [Edge(0, 8), Edge(8, 7), Edge(7, 6), Edge(6, 3)]

**ACTION 3**  
source node: 0  
destination node: 4  
path: [Edge(0, 1), Edge(1, 5), Edge(5, 4)]

**ACTION 4**  
source node: 0  
destination node: 4  
path: [Edge(0, 8), Edge(8, 7), Edge(7, 6), Edge(6, 3), Edge(3, 4)]

**ACTION 5**  
source node: 7  
destination node: 3  
path: [Edge(7, 1), Edge(1, 2), Edge(2, 3)]

**ACTION 6**  
source node: 7  
destination node: 3  
path: [Edge(7, 6), Edge(6, 3)]

**ACTION 7**  
source node: 7  
destination node: 4  
path: [Edge(7, 1), Edge(1, 5), Edge(5, 4)]

**ACTION 8**  
source node: 7  
destination node: 4  
path: [Edge(7, 6), Edge(6, 3), Edge(3, 4)]

### Reward Function

The goal is to minimize the amount of blocking. As a result, a reward of `+1` is given for successfully finding a path that has an available color. On the other hand, if an action is taken and no available color is found, then the reward is `-1`. Finally, if an action is taken and it does not match the incoming request (For example, the incoming request is src=0 dest=4, but the action taken is for src=7 dest=3) then the reward is `-1`.

## Training Setup

### How We Trained the Agent

### Systematic Hyperparam Tuning

## Results

### Learning Curve (Averaged Episode Rewards vs Episode) for Capacity=20

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Episode_Rewards)_for_Capacity_20.png
)
Initially, the averaged episode rewards started at ~58000 and showed gradual growth. It eventually peaked by episode 4 at ~71000 averaged rewards. After this point, the averaged rewards starts to slowly decline. The reason for this curve is most likely because the agent quickly learns which actions to take early on, and then it rapidly starts to gain rewards. Then based on these rewards, the agent knows what actions to take and is less likely to explore for better actions. As it begins to stick with what it thinks is the best action rather than looking for a better action, the average rewards begins to decline.

### Learning Curve (Averaged Objective *B* vs Episode) for Capacity=20

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Objective_B)_for_Capacity_20.png)
This plot correctly shows the inverse of the previous plot. As the agent is exploring for the best actions early on, the initial averaged objective *B* is very high at 0.21. Gradually the agent learns better actions to take and the average objective *B* declines to a minimum of ~0.145 by episode 4. From that point forward, the averaged objective *B* begins to increase again as the agent is no longer looking for better actions.

### Averaged Objective *B* vs Episode on Eval Dataset for Capacity=20

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Objective_B)_for_Capacity_20_EVAL.png)
This plot shows that, on average, objective *B* is between 0.0 and 0.010. There are intermittent spikes where objective *B* rises to 0.015 or even 0.025, but these are few and far between. These results from the evaluation show that the agent is able to consistently assign paths in a manner that minimizes objective *B* when the capacity = 20.

### Learning Curve (Averaged Episode Rewards vs Episode) for Capacity=10

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Episode_Rewards)_for_Capacity_10.png)
The learning curve initially starts at ~5000 averaged rewards and slowly grows. It begins to plateau at episode 3 where the averaged episode rewards is still increasing, but at a much slower rate. One noticeable fact is that the maximum episode rewards here is ~45000 whereas for capactity=20 it was ~71000. This is natural as there is less exploration to be done with less capacity, and thus less rewards to be gained. The agent quickly learns the optimal routing strategy and then the averaged rewards plateaus.

### Learning Curve (Averaged Objective *B* vs Episode) for Capacity=10

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Objective_B)_for_Capacity_10.png)
The averaged objective *B* is initially at a very high ~0.48. This is to be expected as it is much more difficult to efficiently route when the capacity is 10 instead of 20. As the agent began to learn, the averaged objective *B* very quickly reached an averaged objective *B* of ~0.25. We can infer that the averaged objective *B* begins to stablize at ~0.25 because the agent learns an optimal routing policy for capacity 10 and it is not able to improve as quickly.

### Averaged Objective *B* vs Episode on Eval Dataset for Capacity=10

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Objective_B)_for_Capacity_10_EVAL.png)
The plot shows that, on average, objective *B* is between 0.02 and 0.06. There are also intermittent spikes where objective *B* rises to 0.09 or falls to 0.0. This means that the agent's routing policy has a consistent objective *B*. Additionally, the averaged objective *B* for capacity 10 is consistently much higher than capacity 20, which is to be expected as it is easier to block.