import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(BASE_DIR, "results_eval")
SAVE_DIR = os.path.join(BASE_DIR, "ROC_spark_full_5_model")
os.makedirs(SAVE_DIR, exist_ok=True)

model_files = {
    "EfficientNet": os.path.join(RESULT_DIR, "spark_full_efficientnet_result.csv"),
    "MobileNet": os.path.join(RESULT_DIR, "spark_full_mobilenet_result.csv"),
    "ResNet": os.path.join(RESULT_DIR, "spark_full_resnet_result.csv"),
    "DenseNet": os.path.join(RESULT_DIR, "spark_full_densenet_result.csv"),
    "GoogleNet": os.path.join(RESULT_DIR, "spark_full_googlenet_result.csv"),
}

def main():
    model_dfs = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            model_dfs[name] = pd.read_csv(path)
        else:
            print(f"Không tìm thấy file: {path}")

    if not model_dfs:
        print("Không có file dữ liệu CSV nào để vẽ.")
        return

    # Lấy danh sách tên các bệnh từ mô hình đầu tiên (bỏ qua dòng 'Trung bình tất cả')
    sample_df = next(iter(model_dfs.values()))
    class_names = sample_df[sample_df["Class"] != "Trung bình tất cả"]["Class"].tolist()

    for class_name in class_names:
        plt.figure(figsize=(8, 6))

        for model_name, df in model_dfs.items():
            row = df[df["Class"] == class_name]
            if row.empty:
                continue
            
            y_true_str = row.iloc[0]["y_true"]
            y_pred_str = row.iloc[0]["y_pred"]
            
            # Bỏ qua nếu dữ liệu trống
            if pd.isna(y_true_str) or pd.isna(y_pred_str):
                continue

            try:
                # Chuyển string chứa list thành mảng list thực tế
                y_true = ast.literal_eval(y_true_str)
                y_pred = ast.literal_eval(y_pred_str)
            except Exception:
                continue

            # Bỏ qua nếu dữ liệu thực tế chỉ có 1 nhãn (không thể vẽ ROC)
            if len(set(y_true)) < 2:
                print(f"Bỏ qua {model_name} - {class_name} do y_true chỉ chứa 1 loại nhãn.")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc(fpr, tpr):.4f})")

        # Vẽ đường chéo ngẫu nhiên
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        
        # Thiết lập các thuộc tính của đồ thị
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {class_name} - Spark Full")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Xử lý tên file an toàn hơn
        safe_class_name = str(class_name).replace(" ", "_").replace("/", "_")
        save_path = os.path.join(SAVE_DIR, f"ROC_{safe_class_name}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Đã lưu biểu đồ: {save_path}")

if __name__ == "__main__":
    main()
