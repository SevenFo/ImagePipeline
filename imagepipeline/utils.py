from typing import Tuple,List, Optional

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Rectangle
import os

os.environ['DISPLAY'] = "109.105.4.86:1.0"

def visualize(
    image: np.ndarray,
    bboxes: Optional[List[List[int]]] = None,
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> None:
    """
    可视化图像、边界框和mask
    
    参数:
    image: RGB格式的输入图像 [H, W, 3]
    bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)
    save_path: 图片保存路径 (None则显示)
    dpi: 输出图像分辨率
    """
    # 创建画布
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Visualization Results", fontsize=16)
    axes = axs.flatten()
    
    # 子图1: 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 子图2: 带边界框的图像
    axes[1].imshow(image)
    if bboxes:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='r', facecolor='none')
            axes[1].add_patch(rect)
    axes[1].set_title("With Bounding Boxes")
    axes[1].axis('off')
    
    # 子图3: 单独mask
    if mask is not None:
        # 生成彩色mask (忽略0背景)
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap('tab10')(obj_id % 10)[:3]  # 使用tab10颜色循环
            colored_mask[mask == obj_id] = np.array(color) * 255
        axes[2].imshow(colored_mask)
        axes[2].set_title("Segmentation Mask")
    else:
        axes[2].imshow(np.zeros_like(image))
        axes[2].set_title("No Mask Available")
    axes[2].axis('off')
    
    # 子图4: 叠加mask的图像
    axes[3].imshow(image)
    if mask is not None:
        # 创建半透明覆盖层
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color = plt.cm.get_cmap('tab10')(obj_id % 10)
            overlay[mask == obj_id] = [*color[:3], 0.5]  # RGBA格式
        
        axes[3].imshow(overlay)
    axes[3].set_title("Image with Mask Overlay")
    axes[3].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
def compute_mask_centers(
    mask: np.ndarray,
    method: str = 'centroid'
) -> List[Tuple[float, float]]:
    """
    计算多目标mask中每个对象的中心点
    
    参数:
    mask: 多目标mask数组，每个像素值为对象ID (0表示背景)
    method: 中心计算方法，可选'centroid'（质心）或'bounding_box'（边界框中心）
    
    返回:
    中心点坐标列表 [(x1, y1), (x2, y2), ...]，按对象ID升序排列
    """
    centers = []
    unique_ids = np.unique(mask)
    
    for obj_id in unique_ids:
        if obj_id == 0:
            continue  # 跳过背景
        
        if method == 'centroid':
            # 计算质心：所有像素坐标的平均值
            y_indices, x_indices = np.where(mask == obj_id)
            cx = np.mean(x_indices)
            cy = np.mean(y_indices)
            
        elif method == 'bounding_box':
            # 计算边界框中心：包围盒的几何中心
            obj_region = (mask == obj_id)
            rows = np.any(obj_region, axis=1)
            cols = np.any(obj_region, axis=0)
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            
        else:
            raise ValueError(f"Unsupported method: {method}. Choose 'centroid' or 'bounding_box'")
        
        centers.append((cx, cy))
    
    return centers
def visualize_centers(
    image: np.ndarray,
    centers: List[Tuple[float, float]],
    save_path: Optional[str] = None,
    dpi: int = 100
) -> None:
    """
    可视化图像及中心点
    
    参数:
    image: RGB格式的输入图像 [H, W, 3]
    centers: 中心点坐标列表 [(x1, y1), (x2, y2), ...]
    save_path: 图片保存路径 (None则显示)
    dpi: 输出图像分辨率
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Image with Object Centers")
    
    # 绘制中心点（使用不同颜色区分对象）
    for idx, (cx, cy) in enumerate(centers):
        color = plt.cm.tab10(idx % 10)  # 使用tab10颜色循环
        plt.scatter(cx, cy, s=120, edgecolors='black', 
                   linewidths=1.5, facecolor=color,
                   marker='o', label=f'Object {idx+1}')
    
    # 添加图例（显示在图像右侧）
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axis('off')
    
    # 调整布局并保存/显示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Center visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[150:250, 150:250] = 1
    test_mask[350:450, 350:450] = 2
    
    # 调用可视化函数
    visualize(test_image, test_bboxes, test_mask, save_path="demo_visualization.png")

    # 创建测试mask
    test_mask = np.zeros((512, 512), dtype=np.uint8)
    test_mask[150:250, 150:250] = 1  # 对象1：100x100正方形
    test_mask[350:450, 350:450] = 2  # 对象2：100x100正方形
    
    # 计算质心中心
    centroid_centers = compute_mask_centers(test_mask, method='centroid')
    print("Centroid Centers:", centroid_centers)  # 应输出 [(199.5, 199.5), (399.5, 399.5)]
    
    # 计算包围盒中心
    bbox_centers = compute_mask_centers(test_mask, method='bounding_box')
    print("BBox Centers:", bbox_centers)          # 应输出 [(199.5, 199.5), (399.5, 399.5)]