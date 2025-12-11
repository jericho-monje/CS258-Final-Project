##  Begin Local Imports
import model.rsaenv as rsaenv
import model.resource as resource

##  Begin Standard Imports
import argparse
# import gymnasium as gym
import torch
import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common import monitor, utils, env_checker
from pathlib import Path
import matplotlib.pyplot as plot

##  Example environment preset
# env:gym.Env = gym.make("LunarLander-v3", render_mode="human")

##  Custom environment
def make_env(link_capacity:int, seed:int) -> monitor.Monitor:
    env:rsaenv.RSAEnv = rsaenv.RSAEnv(req_file=str(resource.TMP_TRAIN_FILE), link_capacity=link_capacity, max_ht=int(resource.config_values.get_option("MAX_HT")))
    env = monitor.Monitor(env=env)
    env.reset(seed=seed)
    return env

##  Typical DQN config provided in template
CONST_PROVIDED_DQN_CONFIG:dict[str:object] = {
    "learning_rate":1e-3,
    "buffer_size":10_000,
    "learning_starts":100,
    "batch_size":64,
    "tau":0.8,
    "gamma":0.99,
    "train_freq":4,
    "target_update_interval":10_000,
    "exploration_fraction":0.2,
    "exploration_final_eps":0.05
}

"""
Returns the average episode rewards.
First episode is not averaged, first 9 episodes are partially averaged, then beyond that the last 10 episodes
are averaged.
"""
def averaged_evaluations(evaluations):
    averaged_eval_list = []

    for idx in range(len(evaluations)):
        if idx == 0:
            average = evaluations[idx]
        elif idx < 10:
            average = sum(evaluations[:idx]) / (idx + 1)
        else:
            average = sum(evaluations[idx - 9:idx + 1]) / 10
        averaged_eval_list.append(average)
    
    return averaged_eval_list

"""
Creates 3 plots for evaluation and saves as a .PNG
"""
def create_plots(average_episodes, average_blocking, average_eval_blocking, capacity):
    plot.plot(range(1, len(average_episodes) + 1), average_episodes, linewidth=3, color="#5D3FD3")
    plot.xlabel("Episode")
    plot.ylabel("Averaged Episode Rewards")
    plot.title(f"Learning Curve (Averaged Episode Rewards vs Episode) for Capacity={capacity}")
    plot.grid(True, alpha=0.5)
    plot.tight_layout()
    plot.savefig(f"Learning_Curve_(Averaged_Episode_Rewards)_for_Capacity_{capacity}", dpi=250)
    plot.close()

    plot.plot(range(1, len(average_blocking) + 1), average_blocking, linewidth=3, color="#800020")
    plot.xlabel("Episode")
    plot.ylabel("Averaged Objective B")
    plot.title(f"Learning Curve (Averaged Objective B vs Episode) for Capacity={capacity}")
    plot.grid(True, alpha=0.5)
    plot.tight_layout()
    plot.savefig(f"Learning_Curve_(Averaged_Objective_B)_for_Capacity_{capacity}", dpi=250)
    plot.close()

    plot.plot(range(1, len(average_eval_blocking) + 1), average_eval_blocking, linewidth=3, color="#004080")
    plot.xlabel("Episode")
    plot.ylabel("Averaged Objective B")
    plot.title(f"Learning Curve (Averaged Objective B vs Episode) for Capacity={capacity} in eval Dataset")
    plot.grid(True, alpha=0.5)
    plot.tight_layout()
    plot.savefig(f"Learning_Curve_(Averaged_Objective_B)_for_Capacity_{capacity}_EVAL", dpi=250)
    plot.close()

##  Generate a DQN model using the custom RSA Environment, then train and save it.
def generate_and_train_rsadqn(model_path:str, link_capacity:int, seed:int, _debug:int=0) -> None:
    ##  Set random seed value
    if _debug:
        print(f"Setting random seed to value `{str(seed)}`...")
    utils.set_random_seed(seed)

    ##  Make custom RSA Gymnasium Environment with method `make_env`
    if _debug:
        print(f"Making custom RSA Gymnasium Environment...")
    env:monitor.Monitor = make_env(link_capacity=link_capacity, seed=seed)

    ##  Check custom RSA Environment
    env_checker.check_env(rsaenv.RSAEnv(req_file=str(resource.TMP_TRAIN_FILE), link_capacity=link_capacity), warn=True)

    ##  Generate the DQN model with the custom environment
    if _debug:
        print(f"Generating DQN model...")
        print(f"\t[Policy]:: {resource.config_values.get_option("MODEL_POLICY")}")
        print(f"\t[Seed]:: {str(seed)}")
        for k,v in CONST_PROVIDED_DQN_CONFIG.items():
            print(f"\t[{k}]:: {v}")
    if _debug:
        print(f"Is CUDA available?  {torch.cuda.is_available()}")
    model:DQN = DQN(
            device=str(resource.config_values.get_option("MODEL_DEVICE")),
            env=env, 
            policy=str(resource.config_values.get_option("MODEL_POLICY")), 
            seed=seed,
            verbose=_debug + 1,
            **CONST_PROVIDED_DQN_CONFIG
        )

    ##  Reset `env`
    if _debug:
        print(f"Resetting environment...")
    env.reset()

    ##  Train model
    total_timesteps:int = int(resource.config_values.get_option("MAX_HT")) * int(resource.config_values.get_option("N_EPISODES"))
    if _debug:
        print(f"Training model...")
        print(f"\t[Timesteps]:: {str(total_timesteps)}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    ##  Save model to path
    if _debug:
        print(f"Saving model...")
    model.save(model_path)
    print(f"Model saved!!  \n\t[Save Path]::{str(model_path)}")

    ##  Close model
    if _debug:
        print(f"Closing model...")
    env.close()

def test_rsadqn(file:str, model_path:str, link_capacity:int, seed:int, _debug:int=0) -> float:
    ##  Setup
    if _debug:
        print(f"Testing model now...")
        print(f"\t[Seed]:: {seed}")
    test_env:rsaenv.RSAEnv = rsaenv.RSAEnv(req_file=file,link_capacity=link_capacity,max_ht=int(resource.config_values.get_option("MAX_HT")))
    obs, info = test_env.reset(seed=seed)
    ep_return:float = 0.0

    ##  Load the custom trained model generated beforehand
    if _debug:
        print(f"Loading model from path {str(model_path)}")
    model=DQN.load(
            path=model_path,
            device=str(resource.config_values.get_option("MODEL_DEVICE"))
        )

    ##  While not terminated or truncated, predict and step the model while adjusting the reward.
    done=False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if _debug >= 3:
            print(f"\t[Observation]:: {str(obs)}")
        if _debug >= 2:
            print(f"\t[Reward]:: {str(reward)}")
            for k,v in info.items():
                print(f"\t[{k}]:: {v}")
        ep_return += reward
        done = terminated or truncated
        if done:
            print(f"Model running complete!")
            print(f"\t[Blocking Rate]:: {str(float(info["blocks"]))} / {str(float(info["requests"]))} = {str(float(info["blocks"] / info["requests"]))}")
    
    print(f"\tTest episode return: {ep_return:.3f}")

    ##  
    if _debug:
        print(f"Closing test environment now...")
    test_env.close()

    return ep_return

def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--debug", type=int, help="Set debug verbosity.  Integer value 0, 1, or 2.")
    argParser.add_argument("--train", type=Path, help="Train a model.  Incompatible with `--eval`.")
    argParser.add_argument("--eval", type=Path, help="Evaluate a model.  Incompatible with `--train`.")
    argParser.add_argument("--linkcapacity", type=int, help="Define an RSA network link capacity parameter.  Updates config file.")
    args = argParser.parse_args()

    try:
        CONST_DEBUG:int = int(args.debug)
    except Exception as e:
        CONST_DEBUG:int = 0
    print(f"[Debug]:: {'ON' if CONST_DEBUG else 'OFF'} ({CONST_DEBUG})")
    CONST_SEED:int = int(resource.config_values.get_option("SEED"))

    CONST_LINK_CAPACITY:int = int(resource.config_values.get_option("LINK_CAPACITY"))
    if args.linkcapacity:
        try:
            CONST_LINK_CAPACITY = int(args.linkcapacity)
        except Exception as e:
            CONST_LINK_CAPACITY = int(resource.config_values.get_option("LINK_CAPACITY"))
        resource.config_values.set_option(str(CONST_LINK_CAPACITY), "LINK_CAPACITY")
        print(f"RSA Network link capacity set to `{CONST_LINK_CAPACITY}`")
    
    if args.train and args.eval:
        raise Exception(f"Cannot train and evaluate at the same time!")
    elif not args.train and not args.eval:
        raise Exception(f"Must choose to either train or evaluate!!")
    # if args.train and not Path(str(args.train)).exists():
    #     raise FileNotFoundError(f"No such `{str(args.train)}` file found!!")
    # if args.eval and not Path(str(args.eval)).exists():
    #     raise FileNotFoundError(f"No such `{str(args.eval)}` file found!!")

    if args.train:
        target_model_path:Path = None
        try:
            target_model_path = validate_model_path(args.train)
        except Exception as e:
            raise e
        
        generate_and_train_rsadqn(model_path=target_model_path, link_capacity=CONST_LINK_CAPACITY, seed=CONST_SEED, _debug=CONST_DEBUG)

    if args.eval:
        target_model_path:Path = None
        try:
            target_model_path = validate_model_path(args.eval)
        except Exception as e:
            raise e
        
        ep_returns:list[float] = []
        for ia in resource.CONST_EVAL_DATA_DIR.glob("*.csv"):
            print(f"Loading test file:\n\t{str(ia)}")
            ep_returns.append(test_rsadqn(file=str(ia), model_path=target_model_path, link_capacity=CONST_LINK_CAPACITY, seed=CONST_SEED + 1, _debug=CONST_DEBUG))
        print(f"ep_return values for 100 eval request files:\n{str(ep_returns)}")

def validate_model_path(tgt_path:str) -> Path:
    result:Path = Path(str(tgt_path).strip())
    if not result.exists() or not result.suffix == ".zip":
        raise Exception("Inputted model file path not valid...")
    return result

if __name__ == "__main__":
    main()