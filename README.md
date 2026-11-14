# EEG 疲劳论文代码整理

本目录是现有项目文件的原样副本，对应论文 `Noise_Aware_Latent_Fatigue_Trajectory_Learning_from_EEG_under_Noisy_Behavioral_Supervision`。未修改算法、数据路径或实验协议，也未包括数据、权重、结果、论文图表或依赖环境锁定文件。

| 文件 | 对应内容 |
|---|---|
| `run_seedvig_comprehensive.py` | SEED-VIG 回归、分类、消融、跨记录及解释性实验的历史入口 |
| `run_myfatigue_classify_full.py` | SFDE 模型比较、消融、跨受试者及解释性实验的历史入口 |
| `nt_ldfe_model.py` | EEG 编码、时序编码与模型模块 |
| `seed_dataloader.py`, `mydata_dataset.py` | 两个数据集读取 |
| `run_paper_figures.py`, `mne_topomap_utils.py` | 历史绘图与头皮图辅助代码；部分绘图入口包含训练或推理 |
| `nt_ldfe_loso.py`, `nt_ldfe_classify_loso.py` | 历史 LOSO 实验代码，与论文中单次跨记录结果并非同一协议 |
| `run_remaining_experiments.py`, `run_interpretability.py`, `run_ablation_importance.py` | 项目中的轨迹诊断及解释性辅助脚本，部分图的确切来源尚未验证 |
| `protocol.py` | 尚未接入训练流程的协议辅助组件，不是正式实验入口 |

## 数据与路径

源脚本混用当前工作目录和脚本目录定位 `dataset`、`outputs`；复制后路径基准已经改变。默认情况下，数据读取器在 `code/dataset/` 查找数据；其他脚本可能从当前工作目录读取数据，并将结果写到 `code/outputs/`。本目录不包含这些输入，独立运行前须检查各脚本路径和输出位置，不应假定复制后可以直接复现。
