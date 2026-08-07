"""
    Why Loguru?

        Python already has "logging" module, but it is not very user-friendly. 
                            "Loguru" is a library that makes logging in Python easier and more convenient.

        Why use Loguru?

            Because; 

                - Cleaner syntax
                - Better formatting
                - File rotation
                - Easier configuration

"""


from loguru import logger
import os

os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/app.log",
    rotation="5 MB",  # Rotate after 5 MB
    retention="10 days",  # Keep logs for 10 days
    level="INFO"
)

