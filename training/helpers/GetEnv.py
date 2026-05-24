from dotenv import dotenv_values
import os


class GetEnv:
    @staticmethod
    def get_env_variables():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(base_dir, "..", "..", ".env")
        return dotenv_values(env_path)
