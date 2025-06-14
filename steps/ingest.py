import pandas as pd
import yaml
import os

class Ingestion:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        # プロジェクトのルートディレクトリを取得
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'config.yml')
        
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_data(self):
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']
        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        return train_data, test_data
