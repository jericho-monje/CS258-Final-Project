##  Begin Standard Imports
import sys, random
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path

CONST_ROOT_DIR:Path = Path.absolute(Path(sys.argv[0]).parent)
# CONST_OUTPUT_DIR:Path = Path.absolute(Path.joinpath(CONST_ROOT_DIR, r".\output"))
# CONST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONST_MODEL_DIR:Path = Path.absolute(Path.joinpath(CONST_ROOT_DIR, r".\model"))

CONST_EVAL_DATA_DIR:Path = Path.absolute(Path.joinpath(CONST_MODEL_DIR, r".\data\eval"))
CONST_TRAIN_DATA_DIR:Path = Path.absolute(Path.joinpath(CONST_MODEL_DIR, r".\data\train"))

# CONST_DQN_MODEL_PATH:Path = Path.absolute(Path.joinpath(CONST_ROOT_DIR, r".\dqn_rsaenv_model"))

def generateTimestamp() -> str:
    result:str = datetime.now().strftime("%d%m%y-%H%M%S")
    return result

def random_training_file(seed:int) -> Path:
    random.seed(seed)
    chosen_training_file:Path = random.choice(list(CONST_TRAIN_DATA_DIR.glob("*.csv")))
    return chosen_training_file

class configValues:
    def __init__(self) -> None:
        self.CONST_CONFIG_FILE:Path = Path.absolute(Path.joinpath(CONST_ROOT_DIR, r".\config.ini"))
        self.__config_parser:ConfigParser = ConfigParser()

        TMP_VALUES_:dict[str:dict[str:str]] = {
            "DEFAULT" : {
                "SEED" : "123",
                "N_EPISODES" : "1000",
                "LINK_CAPACITY" : "10",
                "MODEL_POLICY" : "MultiInputPolicy",
                "MAX_HT" : "100",
                "MODEL_DEVICE" : "CPU",
                "OPTUNA_TRIALS" : "5"
            }
        }

        if Path.exists(self.CONST_CONFIG_FILE):
            self.__config_parser.read(str(self.CONST_CONFIG_FILE))

            for section,group in TMP_VALUES_.items():
                for item,value in group.items():
                    if not self.__config_parser.has_option(section, item)   \
                        or self.__config_parser[section][item] == "":
                            self.__config_parser.set(section, item, value)

        else:
            for section,group in TMP_VALUES_.items():
                for item,value in group.items():
                    self.__config_parser.set(section, item, value)

        self.__update_config_file()

    def __update_config_file(self) -> None:
        with open(self.CONST_CONFIG_FILE, "w") as f:
            self.__config_parser.write(f, space_around_delimiters=False)

    def get_option(self, option:str, section:str="DEFAULT") -> str:
        self.__config_parser.read(self.CONST_CONFIG_FILE)
        return self.__config_parser[section][option]
    
    def set_option(self, target:str, option:str, section:str="DEFAULT") -> str:
        self.__config_parser.set(section, option, str(target))
        self.__update_config_file()

config_values = configValues()