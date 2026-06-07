# Group13_Happywhale - Whale and Dolphin Identification_111550202

---

## Folder Overview

This folder contains configuration files in `.yaml` format. These files define model-specific hyperparameters such as batch size, input resolution, learning rate, and training schedule.

Each configuration file corresponds to a specific model variant used in experimentation and training.

---

## Configuration Files

### EfficientNet-B6 Additional Configurations 

Resume training after B6 (1024x1024, 30 epochs)

- `efficientnet_b6_824_continue` – Continued training at 824×824 resolution  
- `efficientnet_b6_1024_continue` – Continued training at 1024×1024 resolution  
- `efficientnet_b6_1280_continue` – Continued training at 1280×1280 resolution  
- `efficientnet_b6_1356_continue` – Continued training at 1356×1356 resolution  
- `efficientnet_b6_1440_continue` – Continued training at 1440×1440 resolution  

### EfficientNet-B7 Configurations
- `efficientnet_b7_768` – Standalone model training at 768×768 resolution  
- `efficientnet_b7_1024_ft` – Resume training after B7 (1024x1024, 30 epochs) at 1024×1024 resolution  
- `efficientnet_b7_1024_resume` – Resume training from previous 1024×1024 checkpoint  
- `efficientnet_b7_1280_ft` – Fine-tuned standalone model at 1280×1280 resolution  
- `efficientnet_b7_1280_resume` – Resume training after B7 (1024x1024, 30 epochs) 1280×1280 resolution  
