import skops.io as sio
import pandas as pd
from train import ChurnModelPipeline # Import class từ file train

if __name__ == "__main__":
    pipeline_file = "churn_pipeline.skops"
    data_file = "Churn_Modelling.txt"
    
    # 2. Load model đã có
    print("Loading pre-trained model...")
    trained_model = sio.load(pipeline_file, trusted=True)
    
    # 3. Load data mới để đánh giá
    obj = ChurnModelPipeline(data_file, "Exited")
    obj.load_and_prepare_data(drop_columns=["RowNumber", "CustomerId", "Surname"], nrows=1000)
    
    # Gán model đã load vào object
    obj.model_pipeline = trained_model
    
    # 4. Chạy các hàm đánh giá và xuất file kết quả
    accuracy, f1 = obj.evaluate_model()
    obj.plot_confusion_matrix()
    obj.save_metrics(accuracy, f1)
    obj.plot_roc_curve()
    print("Evaluation complete. Metrics and plots generated.")