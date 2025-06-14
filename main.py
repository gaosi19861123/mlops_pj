import logging
import yaml
import mlflow
import mlflow.sklearn
import os
import tempfile
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report
import click

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

@click.group()
def cli():
    pass

@cli.command()
def train():
    # Load data
    ingestion = Ingestion()
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed successfully")

    # Clean data
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed successfully")

    # Prepare and train model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed successfully")

    # Evaluate model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
    logging.info("Model evaluation completed successfully")
    
    # Print evaluation results
    print("\n============= Model Evaluation Results ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
    print(f"\n{class_report}")
    print("=====================================================\n")

@cli.command()
def train_with_mlflow():
    # macOS用のMLflow設定
    project_dir = os.path.abspath(os.getcwd())
    mlruns_dir = os.path.join(project_dir, "mlruns")
    
    # MLflowの環境変数を設定
    os.environ['MLFLOW_TRACKING_URI'] = f"file://{mlruns_dir}"
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = mlruns_dir
    
    # MLflowディレクトリを作成
    os.makedirs(mlruns_dir, exist_ok=True)
    
    # MLflowの設定
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        train, test = ingestion.load_data()
        logging.info("Data ingestion completed successfully")

        # Clean data
        cleaner = Cleaner()
        train_data = cleaner.clean_data(train)
        test_data = cleaner.clean_data(test)
        logging.info("Data cleaning completed successfully")

        # Prepare and train model
        trainer = Trainer()
        X_train, y_train = trainer.feature_target_separator(train_data)
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed successfully")
        
        # Evaluate model
        predictor = Predictor()
        X_test, y_test = predictor.feature_target_separator(test_data)
        accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
        logging.info("Model evaluation completed successfully")
        
        # Tags 
        mlflow.set_tag('Model developer', 'prsdm')
        mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')
        mlflow.set_tag('platform', 'macOS')
        
        # Log metrics
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)
        model_params = config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc", roc_auc_score)
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        
        # 一時ディレクトリを使用してモデルを保存
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # MLflowにモデルをログ（登録なし）
                mlflow.sklearn.log_model(
                    sk_model=trainer.pipeline,
                    artifact_path="model"
                )
                
                # 手動でモデル情報を記録
                mlflow.log_text(f"Model: {trainer.model_name}", "model_info.txt")
                mlflow.log_text(f"Accuracy: {accuracy:.4f}", "accuracy.txt")
                mlflow.log_text(f"ROC AUC: {roc_auc_score:.4f}", "roc_auc.txt")
                
                logging.info("MLflow tracking completed successfully")
                
            except Exception as e:
                logging.warning(f"MLflow model logging failed: {e}")
                logging.info("Continuing without MLflow model logging...")

        # Print evaluation results
        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
        print(f"\n{class_report}")
        print("=====================================================\n")
        
if __name__ == "__main__":
    cli()
