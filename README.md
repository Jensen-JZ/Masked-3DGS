# Masked-3DGS: Selective 3D Scene Reconstruction with 2D Masks

This repository contains **Masked-3DGS**, an extension of the original **3D Gaussian Splatting** project. The core innovation of this work is the ability to perform **selective 3D scene reconstruction based on 2D masks**. This powerful feature allows the model to focus its attention and computational resources exclusively on user-defined regions of interest, effectively ignoring irrelevant parts of the scene.

This is achieved through three key modifications to the original codebase:

1.  **Modified CUDA Rasterizer**: We've replaced the original rasterizer with a version that outputs the alpha channel (transparency) of the rendered image. This is crucial for comparing the model's output against the input masks during training.
2.  **Mask Data Pipeline**: A new data loading pipeline has been introduced. By using the `--masks <path_to_masks>` command-line argument, you can now provide a directory of binary masks that correspond to your input images. The system will automatically load and utilize these masks during the training process.
3.  **Alpha Loss Function**: We have incorporated a new loss term, controlled by the hyperparameter `--lambda_alpha`. This loss function penalizes any reconstruction occurring outside the masked areas by encouraging the alpha values in the "ignore" regions of the rendered output to be zero.

---

## License

This software is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. For more details, please see the [LICENSE.md](LICENSE.md) file. This means it is available for **non-commercial, research, and evaluation purposes only**.

---

## Usage

Follow these steps to get started with Masked-3DGS:

### 1. Clone the Repository
First, clone the repository and its submodules:
```bash
git clone https://github.com/Jensen-JZ/Masked-3DGS --recursive
```

### 2. Set Up the Environment
We recommend using Conda to manage the environment. Create and activate the environment using the provided file:
```bash
conda env create --file environment.yml
conda activate masked_gs
```

### 3. Prepare Your Data
-   **COLMAP Data**: Ensure your scene data is processed by COLMAP and structured correctly.
-   **Masks**: Create a directory (e.g., `<path_to_your_data>/masks`) and place your mask images inside. Each mask should be a grayscale image where white indicates the region to reconstruct, and its filename must match the corresponding input image.

A standard file structure should look like this:
```
<your_project_directory>/
|---images/
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---masks/
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---sparse/
    |---0/
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

### 4. Run Training
Start the training process by pointing to your dataset and the newly created masks directory. Adjust `--lambda_alpha` as needed.
```bash
python train.py -s <path_to_colmap_dataset> --masks <path_to_mask_directory> --lambda_alpha 0.1
```

---

## Quick Start: Training with Masks

To train a 3DGS model using masks, you first need to prepare a directory containing your mask images. These masks should be grayscale, where white pixels (or values close to 1) represent the region of interest, and black pixels (or values close to 0) represent areas to be ignored. The filenames of the masks must correspond to the filenames of your input images.

Then, you can run the training script with the following new arguments:

```bash
python train.py -s <path_to_colmap_dataset> --masks <path_to_mask_directory> --lambda_alpha 0.1
```

---

## In-depth Parameter Guide

### `--masks`
-   **Usage**: `--masks <path>`
-   **Description**: Specifies the path to the directory containing the mask images. The pipeline will automatically search this directory for mask files that match the names of the input images.

### `--lambda_alpha`
-   **Usage**: `--lambda_alpha <weight>`
-   **Description**: This is a critical hyperparameter that controls the weight of the alpha loss. The choice of this value directly influences how strictly the model adheres to the provided masks. Finding the right value often requires some experimentation based on your specific scene and goals.
-   **Value Ranges and Their Effects**:
    -   **Low Values (e.g., 0.01 - 0.1)**: This provides a gentle constraint. It's useful when you want a softer transition at the mask edges or need to preserve delicate, semi-transparent structures. However, it may not be strong enough to eliminate all "floaters" or unwanted artifacts outside the masked region.
    -   **Medium Values (e.g., 0.1 - 1.0)**: This range typically offers a good balance between quality and mask adherence. It is a strong constraint that is effective at removing most floaters and noise outside the region of interest while generally maintaining good reconstruction quality at the boundaries. A value of `0.1` is a recommended starting point.
    -   **High Values (e.g., > 1.0)**: This imposes a very strict constraint. It will aggressively force the reconstruction to be confined within the mask. While effective, this can sometimes lead to overly sharp or unnatural-looking edges at the mask boundaries, and may even degrade the quality of the reconstruction if the constraint is too harsh.

---

## About The Original 3D Gaussian Splatting Project

This project is built directly upon the groundbreaking research by Kerbl et al. For detailed information about the original 3D Gaussian Splatting method, including environment setup, prerequisites, and advanced usage, please refer to the [Original README file](README_original_3dgs.md).

---
## Acknowledgements

This work is built upon the incredible research and open-source code of the original **3D Gaussian Splatting** project. We extend our sincere gratitude to the authors for their foundational contributions to the field.

-   **Project**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
-   **Repository**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

<br>
<hr>
<br>

# Masked-3DGS: 基于 2D 蒙版的 3D 场景选择性重建

本代码库是 **Masked-3DGS**，一个对原始 **3D Gaussian Splatting** 项目的扩展。本工作的核心创新在于实现了**基于 2D 蒙版的选择性 3D 场景重建**。这一强大功能使得模型能够将其注意力和计算资源完全集中在用户定义的目标区域，从而高效地忽略场景中的无关部分。

该功能主要通过对原始代码库的三项关键修改来实现：

1.  **修改版 CUDA 光栅化器**：我们替换了原始的光栅化器。新版本能够输出渲染图像的 alpha 通道（透明度信息），这对于在训练过程中将模型渲染结果与输入蒙版进行比较至关重要。
2.  **蒙版数据通路**：我们引入了新的数据加载流程。通过使用 `--masks <蒙版路径>` 命令行参数，您现在可以提供一个与输入图像相对应的二进制蒙版目录。系统会自动加载这些蒙版，并在训练过程中使用它们。
3.  **Alpha 损失函数**：我们增加了一个新的损失项，由超参数 `--lambda_alpha` 控制。该损失函数会惩罚在蒙版区域之外进行重建的行为。具体来说，它会促使渲染结果中对应于蒙版“忽略”区域的 alpha 值为零。

---

## 许可

本软件采用“知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议”进行许可。更多详情，请参阅 [LICENSE.md](LICENSE.md) 文件。这意味着本软件**仅可用于非商业、研究和评估目的**。

---

## 使用方法

请遵循以下步骤来开始使用 Masked-3DGS：

### 1. 克隆代码库
首先，克隆本代码库及其所有子模块：
```bash
git clone https://github.com/Jensen-JZ/Masked-3DGS --recursive
```

### 2. 配置环境
我们推荐使用 Conda 来管理项目环境。请使用我们提供的文件来创建并激活 Conda 环境：
```bash
conda env create --file environment.yml
conda activate masked_gs
```

### 3. 准备数据
-   **COLMAP 数据**: 请确保您的场景数据已经由 COLMAP 处理，并具有正确的目录结构。
-   **蒙版**: 创建一个目录（例如 `<您的数据路径>/masks`），并将您的蒙版图像放入其中。每一张蒙版都应为灰度图，其中白色代表需要重建的区域。蒙版的文件名必须与对应的输入图像文件名一致。

一个标准的文件结构应如下所示：
```
<您的项目目录>/
|---images/
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---masks/
|   |---<image_0.png>
|   |---<image_1.png>
|   |---...
|---sparse/
    |---0/
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

### 4. 运行训练
启动训练时，请指定您的数据集路径和新建的蒙版目录路径，并根据需要调整 `--lambda_alpha` 的值。
```bash
python train.py -s <COLMAP 数据集路径> --masks <蒙版目录路径> --lambda_alpha 0.1
```

---

## 快速开始：使用蒙版进行训练

要使用蒙版训练 3DGS 模型，您首先需要准备一个包含蒙版图像的目录。这些蒙版应为灰度图，其中白色像素（或值接近 1）代表感兴趣的区域，而黑色像素（或值接近 0）则代表应被忽略的区域。蒙版的文件名必须与您的输入图像文件名相对应。

然后，您可以使用以下新增的参数来运行训练脚本：

```bash
python train.py -s <COLMAP 数据集路径> --masks <蒙版目录路径> --lambda_alpha 0.1
```

---

## 参数详解

### `--masks`
-   **用法**: `--masks <路径>`
-   **说明**: 指定包含蒙版图像的目录路径。数据加载器会自动在此目录中查找与输入图像同名的蒙版文件。

### `--lambda_alpha`
-   **用法**: `--lambda_alpha <权重值>`
-   **说明**: 这是一个关键的超参数，用于控制 alpha 损失的权重。该值的选择直接影响模型对蒙版约束的严格程度。根据您的具体场景和目标，找到最佳值通常需要一些实验。
-   **不同取值范围及其效果**:
    -   **较低的值 (例如 0.01 - 0.1)**: 提供一个温和的约束。当您希望在蒙版边缘有更平滑的过渡，或者需要保留精细的半透明结构时，这个范围会很有用。但它可能不足以完全消除蒙版外的所有“浮游物”或伪影。
    -   **中等的值 (例如 0.1 - 1.0)**: 通常能提供一个在质量和蒙版遵循度之间的良好平衡。这是一个强约束，能有效去除感兴趣区域外的大部分浮游物和噪声，同时在边界处保持较好的重建质量。我们推荐从 `0.1` 开始尝试。
    -   **较高的的值 (例如 > 1.0)**: 施加一个非常严格的约束。它会极力地将重建限制在蒙版内部。虽然效果显著，但这有时可能导致蒙版边界出现过于生硬、不自然的截断，如果约束过强甚至可能损害重建质量。

---

## 关于原始 3D Gaussian Splatting 项目

本项目直接构建于 Kerbl 等人开创性的研究之上。关于原始 3D Gaussian Splatting 方法的详细信息，包括环境设置、先决条件和高级用法，请参阅[原始 README 文件](README_original_3dgs.md)。

---
## 鸣谢

本工作建立在 **3D Gaussian Splatting** 项目卓越的研究和开源代码基础之上。我们向原作者们为该领域做出的奠基性贡献表示诚挚的感谢。

-   **项目主页**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
-   **代码仓库**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
