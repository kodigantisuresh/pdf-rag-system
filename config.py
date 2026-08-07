from dotenv import load_dotenv
import os

# load_dotenv() ---> Python cannot automatically read the .env file, so we need to load it manually
load_dotenv()

# os.getenv --> Security, if the variable is not found, it will return None instead of throwing an error
# Configuration should not be hard-coded.

APP_NAME = os.getenv("APP_NAME") 
APP_VERSION = os.getenv("APP_VERSION")
LOG_LEVEL = os.getenv("LOG_LEVEL")

