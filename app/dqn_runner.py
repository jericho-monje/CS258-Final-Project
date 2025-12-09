##  Begin Local Imports
import model.rsaenv as rsaenv
import model.resource as resource

##  Begin Standard Imports
# import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import DQN
from stable_baselines3.common import monitor, utils, env_checker

##  Example environment preset
# env:gym.Env = gym.make("LunarLander-v3", render_mode="human")

##  Custom environment
def make_env(seed: int=100) -> monitor.Monitor:
    env:rsaenv.RSAEnv = rsaenv.RSAEnv(req_file=str(resource.TMP_REQ_FILE), max_ht=int(resource.config_values.get_option("max_ht")))
    env = monitor.Monitor(env=env)
    env.reset(seed=seed)

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

##  Generate a DQN model using the custom RSA Environment, then train and save it.
def generate_and_train_rsadqn(seed:int, _debug:int=0) -> None:
    ##  Set random seed value
    if _debug:
        print(f"Setting random seed to value `{str(seed)}`...")
    utils.set_random_seed(seed)

    ##  Generate custom RSA Gymnasium Environment with method `make_env`
    if _debug:
        print(f"Making custom RSA Gymnasium Environment...")
        print(f"\t[Policy]:: {resource.config_values.get_option("MODEL_POLICY")}")
        print(f"\t[Seed]:: {str(seed)}")
        for k,v in CONST_PROVIDED_DQN_CONFIG.items():
            print(f"\t[{k}]:: {v}")
    env:monitor.Monitor = make_env(seed)
    env_checker.check_env(rsaenv.RSAEnv(req_file=str(resource.TMP_REQ_FILE)), warn=True)
    model:DQN = DQN(
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
    model.save(resource.CONST_DQN_MODEL_PATH)
    print(f"Model saved!!  \n\t[Save Path]::{str(resource.CONST_DQN_MODEL_PATH)}.zip")

    ##  Close model
    if _debug:
        print(f"Closing model...")
    env.close()

def test_rsadqn(seed:int, _debug:int=0) -> None:
    ##  Setup
    if _debug:
        print(f"Testing model now...")
        print(f"\t[Seed]:: {seed}")
    test_env:rsaenv.RSAEnv = rsaenv.RSAEnv()
    obs, info = test_env.reset(seed=seed)
    ep_return = 0.0

    ##  Load the custom trained model generated beforehand
    if _debug:
        print(f"Loading model from path {str(resource.CONST_DQN_MODEL_PATH)}.zip")
    model=DQN.load(resource.CONST_DQN_MODEL_PATH)

    ##  While not terminated or truncated, predict and step the model while adjusting the reward.
    done=False
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if _debug:
            print(f"\t[Reward]:: {str(reward)}")
        ep_return += reward
        done = terminated or truncated
        if done:
            print(f"Model running complete!")

    ##  
    if _debug:
        print(f"Closing test environment now...")
    test_env.close()

def main() -> None:
    CONST_DEBUG:int = int(resource.config_values.get_option("DEBUG"))
    CONST_SEED:int = int(resource.config_values.get_option("SEED"))

    generate_and_train_rsadqn(seed=CONST_SEED, _debug=CONST_DEBUG)

    test_rsadqn(seed=CONST_SEED + 1, _debug=CONST_DEBUG)

if __name__ == "__main__":
    main()