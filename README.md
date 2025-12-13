# CS258-Final-Project

## How to Execute

1) **Setup**
    - Install the required Python libraries using the included `requirements.txt` file, optionally using an environment such as `venv` or `conda`.  
    - Using the command line, navigate to the main project folder just above the `app` folder.
    - You may choose to first run `python app/model/resource.py` to fully set up the configuation files.
    - The resulting `app/config.ini` file allows the user to modify some running parameters.  Most of these do not need to be modified for general usage.
2)  **Usage**
    - `app/dqn_runner.py` hosts the main interface for usage via the command line.  
        - To train a model, use the `--train` argument followed by the path to where the model should be stored.  Please include a `.zip` extension.  
           - Use the `--use_tuning` flag to enable Optuna-based hyperparameter-tuning.  
        - To evaluate a model, use the `--eval` argument followed by the path to where the model is stored. 
        - Use the `--link_capacity` argument followed by an integer to set how many frequencies a single link should be able to serve.  
            - *Warning:*  Users may find that the environment updates before the configuration does, resulting in an error regarding the Observation space's link capacity shape and the inputted one.  If this happens, simply repeat the command again.  

3)  **Examples:**

    - **Training for link capacity of 20 with tuning**
        
          python app/dqn_runner.py --train=app/dqn_rsaenv_model_cap20_tuned.zip --use_tuning --link_capacity=20

    - **Evaluation for link capacity of 20**

          python app/dqn_runner.py --eval=app/dqn_rsaenv_model_cap20_tuned.zip --link_capacity=20 


## Requirements
This code has been optimized for Windows computers, but is largely compatible with Linux computers as well.  

A minimum of __Python v3__ is required.  The required "pip" libraries have been listed in `requirements.txt` as well.  

## Environment

### State Representation & State Transitions

In our implementation, we used the generated `nx.Graph` as a topological map.  We did not directly store the states within each LinkState of this graph as we wanted to utilize the more efficient nature of numpy arrays.  The given graph is used to carry data such as action paths, number of links, and so on.  A corresponding `_linkstates` list contained `numpy.ndarray` objects that emulated the states of each link, along with their holding capacity.  Each slot held the holding-time of that particular request.  Slot occupation was assumed when a holding time is above 0.  

The observation state is a dictionary that holds an exploded view of the aforementioned `_linkstates` list.  This showed the slots of each individual link as a multibinary array, forgoing the holding-time and simply showing occupation.  It also contained the details of the current request: source, destination, and holding-time.  

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

Given 10,000 training files, we originally intended to train a singular model on multiple files at once.  However we decided against that and used a random file selection function based on the defined `seed` variable.  This was because we ended up relying on Hyperparam Tuning via Optuna and wanted to maintain a clean learning phase for the model.  The model learns over a number of timesteps that is a high multiple of the `max_ht` variable.

### Systematic Hyperparam Tuning

Our implementation of Hyperparamter-Tuning utilized Optuna's trials via evaluation callbacks within `dqn_runner.py`  Although a simple implementation, it showed effective results in tuning hyperparameters such as the learning rate, gamma, tau, etc.  This was visualized in our graphs, showcasing the progressive learning curves of the model.  Our callback implementation relied on `stable_baselines3.common.callbacks.EvalCallback` and lets the Optuna `Trial` decide how to adjust the parameters upon each step.   The model is created using `optimize_dqn_model()` inside of `objective_function()`.  The learning is then facilitated by `optuna_study_rsadqn()`, which optimizes the hyperparameters over a number of trials we defaulted to 5.  

## Results

### Learning Curve (Averaged Episode Rewards vs Episode) for Capacity=20

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Episode_Rewards)_for_Capacity_20.png
)
Initially, the averaged episode rewards started at ~58000 and showed gradual growth. It eventually peaked by episode 4 at ~71000 averaged rewards. After this point, the averaged rewards starts to slowly decline. The reason for this curve is most likely because the agent quickly learns which actions to take early on, and then it rapidly starts to gain rewards. Then based on these rewards, the agent knows what actions to take and is adapts more slowly to varying situations. As it begins to stick with what it thinks is the best action rather than looking for a better action, the average rewards can decline when encountering episodes with different needs.  This demonstrates the need for data curation/variety to combat "overfitting".

### Learning Curve (Averaged Objective *B* vs Episode) for Capacity=20

![](https://github.com/jericho-monje/CS258-Final-Project/raw/main/Learning_Curve_(Averaged_Objective_B)_for_Capacity_20.png)
This plot correctly shows the inverse of the previous plot. As the agent is exploring for the best actions early on, the initial averaged objective *B* is very high at 0.21. Gradually the agent learns better actions to take and the average objective *B* declines to a minimum of ~0.145 by episode 4. From that point forward, the averaged objective *B* begins to increase again as the agent is less likely to look for better actions.

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