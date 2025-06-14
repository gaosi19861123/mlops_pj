import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self):
        self.model_path = self.load_config()['model']['store_path']
        self.pipeline = self.load_model()

    def load_config(self):
        import yaml
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        return joblib.load(model_file_path)

    def feature_target_separator(self, data):
        # 欠損値のチェック
        if data.iloc[:, -1].isnull().any():
            print("警告: ターゲット変数に欠損値が存在します")
            # 欠損値を持つ行を削除
            data = data.dropna(subset=[data.columns[-1]])
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # 欠損値が完全に除去されたことを確認
        assert not y.isnull().any(), "ターゲット変数に欠損値が残っています"
        
        return X, y

    def evaluate_model(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        return accuracy, class_report, roc_auc
