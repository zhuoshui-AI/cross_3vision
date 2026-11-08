---
license: bsd-3-clause-clear
base_model: facebook/dinov2-base
tags:
- image-feature-extraction
- thermal-imaging
- computer-vision
- knowledge-distillation
- dinov2
- robotics
- multi-modal
datasets:
- theairlabcmu/TartanRGBT
- xjh19972/boson-nighttime
pipeline_tag: image-feature-extraction
library_name: transformers
---

# AnyThermal: Towards Learning Universal Representations for Thermal Perception

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2602.06203-b31b1b.svg)](https://arxiv.org/abs/2602.06203)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://anythermal.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/castacks/AnyThermal)
[![HF Dataset](https://img.shields.io/badge/🤗-TartanRGBT_Dataset-yellow)](https://huggingface.co/datasets/theairlabcmu/TartanRGBT)

</div>

## Model Description

**AnyThermal** is a task-agnostic thermal feature extraction backbone that provides robust representations across diverse environments and robotic perception tasks. Unlike existing thermal models trained on task-specific, small-scale data, AnyThermal generalizes across multiple environments (indoor, aerial, off-road, urban) and tasks without requiring task-specific fine-tuning.

### Key Innovation

AnyThermal distills knowledge from the DINOv2 visual foundation model into a thermal encoder using diverse RGB-Thermal paired data across multiple environments. This approach enables the model to learn universal thermal representations that transfer effectively to downstream tasks.

### Architecture

- **Base Model**: DINOv2 ViT-B/14 (Vision Transformer Base, patch size 14)
- **Parameters**: 86.6M
- **Training Strategy**: Knowledge distillation from frozen RGB DINOv2 teacher to trainable thermal student
- **Input**: Thermal images (converted to 3-channel for compatibility)
- **Output**: 768-dimensional feature embeddings per patch + CLS token

## Training Details

### Knowledge Distillation Process

AnyThermal uses a teacher-student distillation framework:

1. **Teacher Network**: Frozen DINOv2-Base pretrained on RGB images
2. **Student Network**: Trainable DINOv2-Base initialized with RGB weights, processes thermal images
3. **Loss Function**: Contrastive loss on CLS token features from corresponding RGB-thermal pairs
4. **Key Insight**: CLS tokens capture global semantics rather than low-level visual features (like color), making them ideal for cross-modal alignment

This approach relaxes the need for perfect pixel-level alignment or precise synchronization, enabling distillation from datasets with approximate correspondences.

### Training Data

AnyThermal was trained on **five diverse RGB-Thermal datasets** spanning multiple environments:

| Environment | Datasets |
|------------|----------|
| **Urban** | VIVID++, STheReO, Freiburg, TartanRGBT | Driving/Walking scenarios with varied lighting and weather on urban roads, campuses and parks|
| **Aerial** | Boson Nighttime Dataset | Elevated perspectives for mapping and surveillance |
| **Indoor** | TartanRGBT | Buildings with diverse thermal signatures |
| **Off-road** | TartanRGBT | Natural terrain with vegetation and obstacles |

**TartanRGBT** is our newly introduced dataset collected using the first open-source platform with hardware-synchronized RGB-Thermal stereo acquisition. It contributes data across indoor, off-road, and urban environments.
The datset can be found here - [TaratnRGBT Dataset](https://huggingface.co/datasets/theairlabcmu/TartanRGBT)
To know more about the paylaod please visit our project page - [Project Page](https://anythermal.github.io/)



## Capabilities & Performance

AnyThermal demonstrates **state-of-the-art or competitive performance** across multiple thermal perception tasks. We have benchmarked its performance on three tasks 

- Cross-Modal Place Recognition (Thermal query → RGB database)
- Thermal Semantic Segmentation
- Monocular Depth Estimation from Thermal

For both quantitative and qualitative results please visit our [Project Page](https://anythermal.github.io .

We are exploring more tasks where the backbone can be leveragead are are looing forard to learn more from the commutniy how they think AnyThermal can push the frontiers of thermal perception.

## Usage

### Basic Feature Extraction

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load the custom processor and model
repo_id = "theairlabcmu/AnyThermal"
processor = AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id)

# Load your image (works with RGB, Grayscale, etc.)
image = Image.open("path_to_your_image.jpg")

# Preprocess: This automatically handles RGB conversion and 
# snaps dimensions to the nearest multiple of 14.
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
```

### Task-Specific Applications

Please visit our [training and evaluation codebase](https://github.com/castacks/AnyThermal) where we show how to use Anytehrmal and use it with 3 different task specific heads. All thrainign and evaluation wer edoen without any task specific finetuning of the backbone weights. 

## Model Strengths

✅ **Task-Agnostic**: Works across multiple downstream tasks without task-specific training  
✅ **Environment-Agnostic**: Generalizes to indoor, outdoor, urban, off-road, and aerial scenarios  
✅ **Cross-Modal**: Enables thermal-to-RGB and RGB-to-thermal applications  
✅ **Efficient**: Single forward pass produces features for multiple tasks  
✅ **Foundation Model Quality**: Leverages DINOv2's strong semantic representations  

## Limitations

⚠️ **Input Format**: Requires thermal images in 3-channel format (grayscale replicated to RGB)  
⚠️ **Data Bias**: Performance may vary on environments not well-represented in training data

## Ablation Studies

For detailed result please see the Scaling graphs on our [Project Page](https://anythermal.github.io/)

### Impact of Training Data Diversity

**Key Finding**: Multi-environment training is critical. Adding TartanRGBT significantly improves performance across all tasks and domains.


### Single Domain vs. Multi-Domain Training

Training on a single environment (e.g., aerial only) introduces domain bias:
- ✓ Improves performance on that specific domain
- ✗ Reduces performance on other domains (urban, indoor, off-road)

**Conclusion**: Multi-domain RGB-thermal data is essential for learning transferable thermal representations.

## Citation

If you use AnyThermal in your research, please cite:

```bibtex
@misc{maheshwari2026anythermallearninguniversalrepresentations,
      title={AnyThermal: Towards Learning Universal Representations for Thermal Perception}, 
      author={Parv Maheshwari and Jay Karhade and Yogesh Chawla and Isaiah Adu and Florian Heisen and Andrew Porco and Andrew Jong and Yifei Liu and Santosh Pitla and Sebastian Scherer and Wenshan Wang},
      year={2026},
      eprint={2602.06203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.06203}, 
}
```

## Related Resources

- **Paper**: [arXiv:2602.06203](https://arxiv.org/abs/2602.06203)
- **Project Website**: [https://anythermal.github.io/](https://anythermal.github.io/)
- **TartanRGBT Dataset**: [HuggingFace Dataset](https://huggingface.co/datasets/theairlabcmu/TartanRGBT)
- **Data Collection Platform**: [GitHub Repository](https://github.com/AnyThermal/tartan_rgbt_ws)
- **Base Model**: [DINOv2-Base](https://huggingface.co/facebook/dinov2-base)

## License

This model is released under the **BSD-3-Clause-Clear License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted at the AirLab, Carnegie Mellon University. The model builds upon the DINOv2 foundation model from Meta AI Research.

## Model Card Contact

For questions, issues, or collaboration inquiries (Hoping this has sparked your interest!!):
- **Email**: parvm@andrew.cmu.edu
- **GitHub Issues**: [AnyThermal Repository](https://github.com/AnyThermal)
- **Project Website**: [https://anythermal.github.io/](https://anythermal.github.io/)

---

*Last Updated: February 2026*