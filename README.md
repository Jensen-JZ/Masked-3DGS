# Masked-3DGS: Selective 3D Scene Reconstruction with 2D Masks

This repository contains **Masked-3DGS**, an extension of the original **3D Gaussian Splatting** project. The core innovation of this work is the ability to perform **selective 3D scene reconstruction based on 2D masks**. This powerful feature allows the model to focus its attention and computational resources exclusively on user-defined regions of interest, effectively ignoring irrelevant parts of the scene.

This is achieved through three key modifications to the original codebase:

1.  **Modified CUDA Rasterizer**: We've replaced the original rasterizer with a version that outputs the alpha channel (transparency) of the rendered image. This is crucial for comparing the model's output against the input masks during training.
2.  **Mask Data Pipeline**: A new data loading pipeline has been introduced. By using the `--masks <path_to_masks>` command-line argument, you can now provide a directory of binary masks that correspond to your input images. The system will automatically load and utilize these masks during the training process.
3.  **Hybrid Loss Function**: We have incorporated a new hybrid loss term, controlled by the hyperparameter `--lambda_mask`. This loss is a blend of two distinct components: an **Alpha Loss**, which ensures that the alpha channel is zero outside the masked region, and an **Outside Mask Loss**, which penalizes the RGB values of any rendered content appearing outside the designated area.

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
Start the training process by pointing to your dataset and the newly created masks directory. Adjust `--lambda_mask` as needed.

#### Experimental: Vehicle Posture Optimization

Masked-3DGS now supports vehicle posture optimization via the `--reorient_colmap` argument. This feature aligns all vehicles in your dataset to a canonical vehicle coordinate system (+Z up, +X forward), making downstream analysis and visualization more consistent.

**Usage Example:**
```bash
python train.py -s <path_to_colmap_dataset> --masks <path_to_mask_directory> --lambda_mask 0.1 --reorient_colmap
```

When enabled, the system automatically backs up the original `sparse` directory to `sparse_original`, then generates a new, posture-optimized `sparse` directory. All vehicles will be rotated to a unified coordinate frame.

**How It Works:**
- We use principal component analysis (PCA) on both camera poses and point cloud distribution to infer the three main axes:
    - **+Z axis**: Always points upward (roof direction), determined by camera and point cloud vertical distribution.
    - **+X axis**: Points forward along the vehicle, inferred from the longest axis of the point cloud and camera forward vectors.
    - **+Y axis**: Determined by the right-hand rule, ensuring orthogonality.
- The rotation matrix is saved to `alignment_matrix.txt` for reference and further analysis.

**Notes:**
- This is an experimental feature and currently supports only certain scenarios. Future versions may switch to a rasterizer engine with native posture optimization support.
- If your rasterizer supports direct coordinate transformation, more flexible posture correction will be possible.

To disable posture optimization, simply omit the `--reorient_colmap` argument.

---

## Quick Start: Training with Masks

To train a 3DGS model using masks, you first need to prepare a directory containing your mask images. These masks should be grayscale, where white pixels (or values close to 1) represent the region of interest, and black pixels (or values close to 0) represent areas to be ignored. The filenames of the masks must correspond to the filenames of your input images.

Then, you can run the training script with the following new arguments:

```bash
python train.py -s <path_to_colmap_dataset> --masks <path_to_mask_directory> --lambda_mask 0.1
```

---

## In-depth Parameter Guide

### `--masks`
-   **Usage**: `--masks <path>`
-   **Description**: Specifies the path to the directory containing the mask images. The pipeline will automatically search this directory for mask files that match the names of the input images.

### `--lambda_mask`
-   **Usage**: `--lambda_mask <weight>`
-   **Description**: This crucial hyperparameter adjusts the strength of the hybrid mask loss. It directly impacts how strictly the model conforms to the provided masks. The optimal value can vary depending on your scene and specific objectives.
-   **Value Ranges and Their Effects**:
    -   **Low Values (e.g., 0.01 - 0.1)**: Offers a mild constraint, ideal for preserving soft edges or semi-transparent details. May not be sufficient to remove all artifacts outside the mask.
    -   **Medium Values (e.g., 0.1 - 1.0)**: Strikes a good balance between mask fidelity and reconstruction quality. This range is effective at eliminating most external noise while maintaining sharp boundaries. A starting value of `0.1` is recommended.
    -   **High Values (e.g., > 1.0)**: Enforces a strict constraint, aggressively confining the reconstruction to the mask. Can sometimes result in overly sharp or artificial-looking edges.


### `--reorient_colmap` (Experimental)
-   **Usage**: `--reorient_colmap`
-   **Description**: When enabled, the system will automatically rotate the COLMAP sparse model to a canonical vehicle coordinate system (+Z up, +X forward), and back up the original data. This is useful for vehicle datasets where posture unification is required.
-   **Principle**: The axes are inferred automatically using principal component analysis (PCA) on both camera poses and point cloud distribution.
-   **Note**: This is an experimental feature. In future versions, posture optimization may be natively supported by switching to a rasterizer engine that allows direct coordinate transformation, enabling more flexible and robust posture correction.

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
3.  **混合损失函数**：我们引入了一个新的混合损失项，由超参数 `--lambda_mask` 控制。该损失函数融合了两个不同的部分：**Alpha 损失**，确保蒙版外的 alpha 通道值为零；以及 **蒙版外损失**，用于惩罚在指定区域外渲染出的任何内容的 RGB 值。

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
启动训练时，请指定您的数据集路径和新建的蒙版目录路径，并根据需要调整 `--lambda_mask` 的值。

#### 实验性：车辆姿态优化

Masked-3DGS 现已支持通过 `--reorient_colmap` 参数对 COLMAP 输出的稀疏模型进行**车身姿态优化**，可将车辆统一对齐到标准车体坐标系（+Z 向上，+X 向前）。

**用法示例：**
```bash
python train.py -s <COLMAP 数据集路径> --masks <蒙版目录路径> --lambda_mask 0.1 --reorient_colmap
```

启用该参数后，系统会自动将原始 `sparse` 目录备份为 `sparse_original`，并生成一个经过旋转优化的新 `sparse`，所有车辆都将对齐到统一的坐标系。

**原理简介：**
- 我们通过主成分分析（PCA）结合相机姿态和点云分布，自动推断车体的三个主轴：
    - **+Z 轴**：始终指向竖直向上（车顶方向），通过相机分布和点云法向自动确定。
    - **+X 轴**：指向车辆前方，结合点云长轴和相机前向自动推断。
    - **+Y 轴**：由右手坐标系自动确定，保证三轴正交。
- 旋转矩阵会被保存到 `alignment_matrix.txt`，便于后续分析。

**注意事项：**
- 该功能为实验性，当前仅支持部分场景，未来可能会更换栅格化引擎以原生支持姿态优化。
- 若后续栅格化器支持直接指定坐标变换，则可实现更灵活的姿态校正。

如需禁用姿态优化，只需不加 `--reorient_colmap` 参数即可。

---

## 快速开始：使用蒙版进行训练

要使用蒙版训练 3DGS 模型，您首先需要准备一个包含蒙版图像的目录。这些蒙版应为灰度图，其中白色像素（或值接近 1）代表感兴趣的区域，而黑色像素（或值接近 0）则代表应被忽略的区域。蒙版的文件名必须与您的输入图像文件名相对应。

然后，您可以使用以下新增的参数来运行训练脚本：

```bash
python train.py -s <COLMAP 数据集路径> --masks <蒙版目录路径> --lambda_mask 0.1
```

---

## 参数详解

### `--masks`
-   **用法**: `--masks <路径>`
-   **说明**: 指定包含蒙版图像的目录路径。数据加载器会自动在此目录中查找与输入图像同名的蒙版文件。

### `--lambda_mask`
-   **用法**: `--lambda_mask <权重值>`
-   **说明**: 这是一个关键的超参数，用于调整混合蒙版损失的强度。它直接影响模型对所提供蒙版的遵循严格程度。最佳值可能因您的场景和具体目标而异。
-   **不同取值范围及其效果**:
    -   **较低的值 (例如 0.01 - 0.1)**: 提供一个温和的约束，非常适合保留柔和的边缘或半透明细节。可能不足以移除蒙版外的所有伪影。
    -   **中等的值 (例如 0.1 - 1.0)**: 在蒙版保真度和重建质量之间取得了良好的平衡。这个范围能有效消除大部分外部噪声，同时保持清晰的边界。建议从 `0.1` 开始。
    -   **较高的的值 (例如 > 1.0)**: 强制执行严格的约束，积极地将重建限制在蒙版内。有时可能导致边缘过于锐利或看起来不自然。

### `--reorient_colmap` (实验性)
-   **用法**: `--reorient_colmap`
-   **说明**: 启用后，系统会自动将 COLMAP 输出的稀疏模型旋转到标准车体坐标系（+Z 向上，+X 向前），并备份原始数据。适用于车辆等需要统一姿态的场景。
-   **原理**: 结合相机分布和点云主成分分析（PCA），自动推断三轴方向。
-   **注意**: 当前为实验性功能，未来可能会通过更换栅格化引擎实现更灵活的姿态优化。

---

## 关于原始 3D Gaussian Splatting 项目

本项目直接构建于 Kerbl 等人开创性的研究之上。关于原始 3D Gaussian Splatting 方法的详细信息，包括环境设置、先决条件和高级用法，请参阅[原始 README 文件](README_original_3dgs.md)。

---
## 鸣谢

本工作建立在 **3D Gaussian Splatting** 项目卓越的研究和开源代码基础之上。我们向原作者们为该领域做出的奠基性贡献表示诚挚的感谢。

-   **项目主页**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
-   **代码仓库**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)