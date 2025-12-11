##  Begin Local Imports
import model.rsaenv as rsaenv
import model.resource as resource

##  Begin Standard Imports
import argparse
import random
# import gymnasium as gym
import torch
import stable_baselines3 as sb3
import optuna
from matplotlib import pyplot as plot
from stable_baselines3 import DQN
from stable_baselines3.common import monitor, utils, env_checker, callbacks, evaluation
from pathlib import Path

##  Example environment preset
# env:gym.Env = gym.make("LunarLander-v3", render_mode="human")

##  Custom environment
def make_env(training_file:str, link_capacity:int, seed:int) -> monitor.Monitor:
    env:rsaenv.RSAEnv = rsaenv.RSAEnv(req_file=training_file, link_capacity=link_capacity, max_ht=int(resource.config_values.get_option("MAX_HT")))
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
            average = sum(evaluations[:idx + 1]) / (idx + 1)
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

class CustomTrialCallback(callbacks.EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=1000, deterministic=True, render=False):
        super(CustomTrialCallback, self).__init__(eval_env, trial, n_eval_episodes, eval_freq, deterministic, render)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward, _ = self.evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, render=self.render, deterministic=self.deterministic)
            self.trial.report(-mean_reward, self.n_calls)

            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return True
    
def optimize_dqn_model(trial:optuna.Trial) -> dict[str, object]:
    ##  Suggest hyperparameters
    learning_rate:float = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size:int = trial.suggest_categorical("buffer_size", [5_000, 10_000, 20_000, 50_000])
    batch_size:int = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma:float = trial.suggest_float("gamma", 0.8, 0.9999, log=True)
    tau:float = trial.suggest_float("tau", 0.5, 0.99)
    train_freq:int = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    target_update_interval:int = trial.suggest_categorical("target_update_interval", [1_000, 5_000, 10_000, 20_000])
    exploration_fraction:float = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps:float = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    ##  Return suggested hyperparameters as dict
    return {
        "learning_rate":learning_rate,
        "buffer_size":buffer_size,
        "batch_size":batch_size,
        "gamma":gamma,
        "tau":tau,
        "train_freq":train_freq,
        "target_update_interval":target_update_interval,
        "exploration_fraction":exploration_fraction,
        "exploration_final_eps":exploration_final_eps
    }

def objective_function(trial:optuna.Trial, link_capacity:int, seed:int, _debug:int=0) -> float:
    target_training_file:Path = resource.random_training_file(seed=seed)
    ##  Make custom RSA Gymnasium Environment with method `make_env`
    env:monitor.Monitor = make_env(training_file=str(target_training_file), link_capacity=link_capacity, seed=seed)
    ##  Check custom RSA Environment
    env_checker.check_env(rsaenv.RSAEnv(req_file=str(target_training_file), link_capacity=link_capacity), warn=True)
    ##  Get suggested hyperparameters from `optimize_dqn_model`
    suggested_params:dict[str, object] = optimize_dqn_model(trial)
    ##  Generate the DQN model with the custom environment
    if _debug:
        print(f"Generating Tuned DQN model...")
        print(f"\t[Policy]:: {resource.config_values.get_option("MODEL_POLICY")}")
        print(f"\t[Seed]:: {str(seed)}")
        for k,v in suggested_params.items():
            print(f"\t[{k}]:: {v}")
    if _debug:
        print(f"Is CUDA available?  {torch.cuda.is_available()}")
    model:DQN = DQN(
            device=str(resource.config_values.get_option("MODEL_DEVICE")),
            env=env, 
            policy=str(resource.config_values.get_option("MODEL_POLICY")), 
            seed=seed,
            verbose=_debug + 1,
            **suggested_params
        )
    ##  Reset `env`
    env.reset()
    ##  Train model
    total_timesteps:int = int(resource.config_values.get_option("MAX_HT")) * int(resource.config_values.get_option("N_EPISODES"))
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    ##  Evaluate model
    mean_reward, _ = evaluation.evaluate_policy(model, env, n_eval_episodes=5, render=False, deterministic=True)
    if _debug:
        print(f"Mean reward over evaluation episodes: {mean_reward}")
    ##  Close model
    env.close()

    return mean_reward

def optuna_study_rsadqn(link_capacity:int, seed:int, n_trials:int, _debug:int=0) -> optuna.Study:
    ##  Create Optuna study
    if _debug:
        print(f"Creating Optuna study for DQN hyperparameter tuning...")
    study:optuna.Study = optuna.create_study(direction="maximize", study_name="DQN_RSA_Hyperparameter_Tuning")
    ##  Optimize study
    if _debug:
        print(f"Optimizing DQN hyperparameters over {str(n_trials)} trials...")
    study.optimize(lambda trial: objective_function(trial, link_capacity, seed, _debug), n_trials=n_trials)
    if _debug:
        print(f"Best trial:")
        trial:optuna.Trial = study.best_trial
        print(f"\tValue: {trial.value}")
        print(f"\tParams: ")
        for k,v in trial.params.items():
            print(f"\t\t{k}: {v}")
    return study

##  Generate a DQN model using the custom RSA Environment, then train and save it.
def generate_and_train_rsadqn(model_path:str, link_capacity:int, seed:int, use_tuning:bool=False, _debug:int=0) -> None:
    ##  Set random seed value
    if _debug:
        print(f"Setting random seed to value `{str(seed)}`...")
    utils.set_random_seed(seed)
    
    target_training_file:Path = resource.random_training_file(seed=seed)

    ##  Make custom RSA Gymnasium Environment with method `make_env`
    if _debug:
        print(f"Making custom RSA Gymnasium Environment...")
    env:monitor.Monitor = make_env(training_file=str(target_training_file), link_capacity=link_capacity, seed=seed)

    ##  Check custom RSA Environment
    env_checker.check_env(rsaenv.RSAEnv(req_file=str(target_training_file), link_capacity=link_capacity), warn=True)

    chosen_dqn_config:dict[str, object] = CONST_PROVIDED_DQN_CONFIG
    if use_tuning:
        if _debug:
            print(f"Optimizing DQN hyperparameters before training...")
        study:optuna.Study = optuna_study_rsadqn(link_capacity=link_capacity, seed=seed, n_trials=int(resource.config_values.get_option("OPTUNA_TRIALS")), _debug=_debug)
        chosen_dqn_config = study.best_trial.params

    ##  Generate the DQN model with the custom environment
    if _debug:
        print(f"Generating DQN model...")
        print(f"\t[Policy]:: {resource.config_values.get_option("MODEL_POLICY")}")
        print(f"\t[Seed]:: {str(seed)}")
        for k,v in chosen_dqn_config.items():
            print(f"\t[{k}]:: {v}")
    if _debug:
        print(f"Is CUDA available?  {torch.cuda.is_available()}")
    model:DQN = DQN(
            device=str(resource.config_values.get_option("MODEL_DEVICE")),
            env=env, 
            policy=str(resource.config_values.get_option("MODEL_POLICY")), 
            seed=seed,
            verbose=_debug + 1,
            **chosen_dqn_config
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

    return 

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
            blocking_rate:float = float(info["blocks"] / info["requests"]) if info["requests"] > 0 else 0.0
            print(f"Model running complete!")
            print(f"\t[Blocking Rate]:: {str(float(info["blocks"]))} / {str(float(info["requests"]))} = {blocking_rate:.3f}")
    
    print(f"\tTest episode return: {ep_return:.3f}")

    ##  
    if _debug:
        print(f"Closing test environment now...")
    test_env.close()

    return ep_return, blocking_rate

def main() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--debug", type=int, help="Set debug verbosity.  Integer value 0, 1, or 2.")
    argParser.add_argument("--train", type=Path, help="Train a model.  Incompatible with `--eval`.")
    argParser.add_argument("--use_tuning", action="store_true", help="Use Optuna hyperparameter tuning when training the model.  Only valid with `--train`.")
    argParser.add_argument("--eval", type=Path, help="Evaluate a model.  Incompatible with `--train`.")
    argParser.add_argument("--link_capacity", type=int, help="Define an RSA network link capacity parameter.  Updates config file.")
    argParser.add_argument("--seed", type=int, help="Define the seed parameter to be used by the model.  Updates config file.")
    
    argParser.add_argument("--test", action="store_true", help="Testing optuna trial.  Not for general use.")
    args = argParser.parse_args()

    try:
        CONST_DEBUG:int = int(args.debug)
    except Exception as e:
        CONST_DEBUG:int = 0
    print(f"[Debug]:: {'ON' if CONST_DEBUG else 'OFF'} ({CONST_DEBUG})")

    CONST_SEED:int = int(resource.config_values.get_option("SEED"))
    if args.seed:
        try:
            CONST_SEED = int(args.seed)
        except Exception as e:
            CONST_SEED = int(resource.config_values.get_option("SEED"))
        resource.config_values.set_option(str(CONST_SEED), "SEED")
        print(f"Seed set to `{CONST_SEED}`")

    CONST_LINK_CAPACITY:int = int(resource.config_values.get_option("LINK_CAPACITY"))
    if args.link_capacity:
        try:
            CONST_LINK_CAPACITY = int(args.link_capacity)
        except Exception as e:
            CONST_LINK_CAPACITY = int(resource.config_values.get_option("LINK_CAPACITY"))
        resource.config_values.set_option(str(CONST_LINK_CAPACITY), "LINK_CAPACITY")
        print(f"RSA Network link capacity set to `{CONST_LINK_CAPACITY}`")
    
    if args.test:
        print(f"Running optuna study test...")
        optuna_study_rsadqn(link_capacity=CONST_LINK_CAPACITY, seed=CONST_SEED, n_trials=int(resource.config_values.get_option("OPTUNA_TRIALS")), _debug=CONST_DEBUG)
        return

    if args.train and args.eval:
        raise Exception(f"Cannot train and evaluate at the same time!")
    elif not args.train and not args.eval:
        raise Exception(f"Must choose to either train or evaluate!!")

    if args.train:
        target_model_path:Path = None
        try:
            target_model_path, target_exists = validate_model_path(args.train)
        except Exception as e:
            raise e
        
        generate_and_train_rsadqn(model_path=target_model_path, link_capacity=CONST_LINK_CAPACITY, seed=CONST_SEED, use_tuning=args.use_tuning, _debug=CONST_DEBUG)
        return

    if args.eval:
        target_model_path:Path = None
        try:
            target_model_path, target_exists = validate_model_path(args.eval)
        except Exception as e:
            raise e
        
        if not target_exists:
            raise Exception(f"Inputted model file path does not exist...")
        
        ep_returns:list[float] = []
        blocking_rates:list[float] = []
        for ia in resource.CONST_EVAL_DATA_DIR.glob("*.csv"):
            print(f"Loading test file:\n\t{str(ia)}")
            ep_return, blocking_rate = test_rsadqn(file=str(ia), model_path=target_model_path, link_capacity=CONST_LINK_CAPACITY, seed=CONST_SEED + 1, _debug=CONST_DEBUG)
            ep_returns.append(ep_return)
            blocking_rates.append(blocking_rate)
        if CONST_DEBUG >= 2:
            print(f"ep_return values for 100 eval request files:\n{str(ep_returns)}")
        episodes = len(ep_returns)
        
        #episodes, ep_returns, blocking_rates

        return

def validate_model_path(tgt_path:str) -> tuple[Path, bool]:
    result:Path = Path(str(tgt_path).strip())
    # if not result.exists() or not result.suffix == ".zip":
    if not result.suffix == ".zip":
        raise Exception("Inputted model file path not valid...")
    return result, Path.exists(result)

if __name__ == "__main__":
    main()