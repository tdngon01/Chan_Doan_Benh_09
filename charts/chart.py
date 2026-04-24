import matplotlib.pyplot as plt
import numpy as np

Model_info = {
        "1.MoCo_LoRA_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_EfficientNet", "lora_finetune_best_auc_0.9260.pth.tar"),
            "type": "lora_moco",
            "backbone": "EfficientNet",
            "save": "moco_lora_efficientnet_result.csv",
        },
        "2.MoCo_Full_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_EfficientNet", "full_finetune_best_auc_0.9264.pth.tar"),
            "type": "full_moco",
            "backbone": "EfficientNet",
            "save": "moco_full_efficientnet_result.csv",
        },
        "3.MoCo_LoRA_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_MobileNet", "lora_finetune_best_auc_0.9212.pth.tar"),
            "type": "lora_moco",
            "backbone": "MobileNet",
            "save": "moco_lora_mobilenet_result.csv",
        },
        "4.MoCo_Full_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_MobileNet", "full_finetune_best_auc_0.9139.pth.tar"),
            "type": "full_moco",
            "backbone": "MobileNet",
            "save": "moco_full_mobilenet_result.csv",
        },
        "5.MoCo_LoRA_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_ResNet", "lora_finetune_best_auc_0.9181.pth.tar"),
            "type": "lora_moco",
            "backbone": "ResNet",
            "save": "moco_lora_resnet_result.csv",
        },
        "6.MoCo_Full_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_ResNet", "full_finetune_best_auc_0.9228.pth.tar"),
            "type": "full_moco",
            "backbone": "ResNet",
            "save": "moco_full_resnet_result.csv",
        },
        "7.MoCo_LoRA_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_DenseNet", "lora_finetune_best_auc_0.9183.pth.tar"),
            "type": "lora_moco",
            "backbone": "DenseNet",
            "save": "moco_lora_densenet_result.csv",
        },
        "8.MoCo_Full_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_DenseNet", "full_finetune_best_auc_0.9267.pth.tar"),
            "type": "full_moco",
            "backbone": "DenseNet",
            "save": "moco_full_densenet_result.csv",
        },
        "9.MoCo_LoRA_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_GoogleNet", "lora_finetune_best_auc_0.9270.pth.tar"),
            "type": "lora_moco",
            "backbone": "GoogleNet",
            "save": "moco_lora_googlenet_result.csv",
        },
        "10.MoCo_Full_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_GoogleNet", "full_finetune_best_auc_0.9201.pth.tar"),
            "type": "full_moco",
            "backbone": "GoogleNet",
            "save": "moco_full_googlenet_result.csv",
        },
        #==============SparK==========================================
        "11.SparK_LoRA_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_SPARK, "lora_finetune_EfficientNet", "lora_finetune_best_auc_0.9259.pth.tar"),
            "type": "lora_spark",
            "backbone": "EfficientNet",
            "save": "spark_lora_efficientnet_result.csv"

        },
        "12.SparK_Full_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_SPARK, "full_finetune_EfficientNet", "full_finetune_best_auc_0.9217.pth.tar"),
            "type": "full_spark",
            "backbone": "EfficientNet",
            "save": "spark_full_efficientnet_result.csv"
        },
        "13.SparK_LoRA_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_SPARK, "lora_finetune_MobileNet", "lora_finetune_best_auc_0.9125.pth.tar"),
            "type": "lora_spark",
            "backbone": "MobileNet",
            "save": "spark_lora_mobilenet_result.csv"
        },
        "14.SparK_Full_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_SPARK, "full_finetune_MobileNet", "full_finetune_best_auc_0.9123.pth.tar"),
            "type": "full_spark",
            "backbone": "MobileNet",
            "save": "spark_full_mobilenet_result.csv"
        },
        "15.SparK_LoRA_ResNet-18": {
            "path": os.path.join(CHECKPOINT_SPARK, "lora_finetune_ResNet", "lora_finetune_best_auc_0.9035.pth.tar"),
            "type": "lora_spark",
            "backbone": "ResNet",
            "save": "spark_lora_resnet_result.csv"
        },
        "16.SparK_Full_ResNet-18": {
            "path": os.path.join(CHECKPOINT_SPARK, "full_finetune_ResNet", "full_finetune_best_auc_0.9245.pth.tar"),
            "type": "full_spark",
            "backbone": "ResNet",
            "save": "spark_full_resnet_result.csv"
        },
        "17.SparK_LoRA_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_SPARK, "lora_finetune_DenseNet", "lora_finetune_best_auc_0.9109.pth.tar"),
            "type": "lora_spark",
            "backbone": "DenseNet",
            "save": "spark_lora_densenet_result.csv"
        },
        "18.SparK_Full_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_SPARK, "full_finetune_DenseNet", "full_finetune_best_auc_0.9223.pth.tar"),
            "type": "full_spark",
            "backbone": "DenseNet",
            "save": "spark_full_densenet_result.csv"
        },
        "19.SparK_LoRA_GoogleNet": {
            "path": os.path.join(CHECKPOINT_SPARK, "lora_finetune_GoogleNet", "lora_finetune_best_auc_0.9207.pth.tar"),
            "type": "lora_spark",
            "backbone": "GoogleNet",
            "save": "spark_lora_googlenet_result.csv"
        },
        "20.SparK_Full_GoogleNet": {
            "path": os.path.join(CHECKPOINT_SPARK, "full_finetune_GoogleNet", "full_finetune_best_auc_0.9182.pth.tar"),
            "type": "full_spark",
            "backbone": "GoogleNet",
            "save": "spark_full_googlenet_result.csv"
        },
    }


fig, ax = plt.subplots()

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()