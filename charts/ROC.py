import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_DIR = os.path.join(BASE_DIR, "results_eval")
SAVE_DIR = os.path.join(BASE_DIR, "ROC_model")
os.makedirs(SAVE_DIR, exist_ok=True)

model_files = {
        "1.MoCo_LoRA_EfficientNet-B0": os.path.join(RESULT_DIR, "moco_lora_efficientnet_result.csv"),
        "2.MoCo_Full_EfficientNet-B0": os.path.join(RESULT_DIR, "moco_full_efficientnet_result.csv"),
        "3.MoCo_LoRA_MobileNet-V2": os.path.join(RESULT_DIR, "moco_lora_mobilenet_result.csv"),
        "4.MoCo_Full_MobileNet-V2": os.path.join(RESULT_DIR, "moco_full_mobilenet_result.csv"),
        "5.MoCo_LoRA_ResNet-18": os.path.join(RESULT_DIR, "moco_lora_resnet_result.csv"),
        "6.MoCo_Full_ResNet-18": os.path.join(RESULT_DIR, "moco_full_resnet_result.csv"),
        "7.MoCo_LoRA_DenseNet-121": os.path.join(RESULT_DIR, "moco_lora_densenet_result.csv"),
        "8.MoCo_Full_DenseNet-121": os.path.join(RESULT_DIR, "moco_full_densenet_result.csv"),
        "9.MoCo_LoRA_GoogleNet": os.path.join(RESULT_DIR, "moco_lora_googlenet_result.csv"),
        "10.MoCo_Full_GoogleNet": os.path.join(RESULT_DIR, "moco_full_googlenet_result.csv"),
        "11.MoCo_LoRA_VGG16": os.path.join(RESULT_DIR, "moco_lora_vgg16_result.csv"),
        "12.MoCo_Full_VGG16": os.path.join(RESULT_DIR, "moco_full_vgg16_result.csv"),
}

def main():
    model_dfs = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            model_dfs[name] = pd.read_csv(path)
        else:
            print(f"Không tìm thấy file: {path}")

    # Lấy danh sách tên các bệnh từ mô hình đầu tiên (bỏ qua dòng 'Trung bình tất cả')
    sample_df = next(iter(model_dfs.values()))
    class_names = sample_df[sample_df["Class"] != "Trung bình tất cả"]["Class"].tolist()

    for model_name, df in model_dfs.items():
        plt.figure(figsize=(8, 8))

        for class_name in class_names:
            row = df[df["Class"] == class_name]
            if row.empty:
                continue
            
            y_true_str = row.iloc[0]["y_true"]# trỏ vào csv
            y_pred_str = row.iloc[0]["y_pred"]

            try:
                y_true = ast.literal_eval(y_true_str)
                y_pred = ast.literal_eval(y_pred_str)

                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                
                # Vẽ từng bệnh
                plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.4f})")
            except Exception as e:
                print(f"Lỗi khi xử lý bệnh {class_name} trong {model_name}: {e}")
                continue

        # Vẽ đường chéo ngẫu nhiên
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        
        # Thiết lập các thuộc tính của đồ thị
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC-AUC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Xử lý tên file
        safe_model_name = model_name.replace(" ", "_").replace(".", "_")
        save_path = os.path.join(SAVE_DIR, f"ROC_{safe_model_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Đã lưu biểu đồ: {save_path}")

if __name__ == "__main__":
    main()
