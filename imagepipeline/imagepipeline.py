from typing import Callable
import torch
import json_repair
import cv2
import numpy as np
import base64
import pyrealsense2 as rs
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from torchvision.transforms.functional import to_tensor
from io import BytesIO

from utils import visualize, compute_mask_centers,visualize_centers
from model_adapters import QwenVLAdapter

class ImagePipeline:
    def __init__(self, sam_checkpoint: str, sam_model_type: str, cutie_model, vl_adapter):
        """
        多目标分割与跟踪管道（集成视觉语言模型）
        
        参数:
        sam_checkpoint: SAM模型权重路径
        sam_model_type: SAM模型类型 (e.g. "vit_h")
        cutie_model: 预加载的CUTIE模型实例
        vl_adapter: 视觉语言模型适配器实例（QwenVLAdapter）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化SAM
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
        # 初始化CUTIE
        self.cutie = cutie_model
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1
        
        # 集成视觉语言模型
        self.vl_adapter = vl_adapter
        
        # 状态跟踪
        self.current_objects = []
        self.is_initialized = False
        
    def _image_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像转换为base64编码字符串"""
        pil_img = Image.fromarray(image)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_bbox_from_vl(self, frame: np.ndarray, instruction: str) -> list:
        """
        使用视觉语言模型生成目标边界框
        
        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        instruction: 自然语言指令（如"the red cup on the left"）
        
        返回:
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        """
        # 转换图像格式
        base64_image = self._image_to_base64(frame)
        
        # 构建VL模型输入
        prompt = (
            f"Analyze the image and identify ALL objects matching: {instruction}.\n"
            "Return bboxes for ALL matching objects in this format:\n"
            "[{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"...\"}, ...]"
        )
        
        print(f"vl user prompt:\n{prompt}")
        
        # 生成响应
        input_data = self.vl_adapter.prepare_input(
            text=prompt,
            image_url=f"data:image/jpeg;base64,{base64_image}"
        )
        response, _ = self.vl_adapter.generate_response(input_data, max_tokens=512)
        
        print(f"response from vl: {response}")
        
        # 解析响应
        try:
            # 提取JSON部分
            json_str = response[response.find("["):response.rfind("]")+1]
            bbox_list = json_repair.loads(json_str)
            
            # 验证bbox格式
            valid_bboxes = []
            for item in bbox_list:
                bbox = item.get("bbox_2d", [])
                if len(bbox) == 4 and all(0 <= v <= frame.shape[1] if i%2==0 else 0 <= v <= frame.shape[0] for i,v in enumerate(bbox)):
                    valid_bboxes.append(bbox)
            return valid_bboxes
        except Exception as e:
            raise RuntimeError(f"Failed to parse VL model response: {str(e)}")

    def initialize_with_instruction(self, frame: np.ndarray, instruction: str, return_bbox: bool = False) -> tuple[np.ndarray,list|None]:
        """
        端到端初始化流程：VL生成bbox -> SAM分割 -> CUTIE初始化
        
        参数:
        frame: RGB格式的输入图像
        instruction: 自然语言指令
        
        返回:
        combined_mask: 组合后的多目标mask
        """
        # Step 1: 通过VL模型获取bbox
        bboxes = self.get_bbox_from_vl(frame, instruction)
        if not bboxes:
            raise ValueError("No valid bounding boxes detected by VL model")
        
        # Step 2: SAM生成mask
        return self.initialize_masks(frame, bboxes), None if not return_bbox else bboxes
    def initialize_masks(self, frame: np.ndarray, bboxes: list) -> np.ndarray:
        """
        初始化多目标分割
        
        参数:
        frame: RGB格式的输入图像 [H, W, 3]
        bboxes: 多个目标的边界框列表 [[x1, y1, x2, y2], ...]
        
        返回:
        combined_mask: 组合后的多目标mask，每个目标用不同整数ID表示
        """
        # 转换颜色空间并设置SAM图像
        rgb_frame = frame
        self.predictor.set_image(rgb_frame)
        
        # 生成并组合多个目标的mask
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        object_ids = []
        for obj_idx, bbox in enumerate(bboxes):
            # SAM预测最佳mask
            masks, scores, _ = self.predictor.predict(
                box=np.array(bbox),
                multimask_output=True
            )
            best_mask = masks[np.argmax(scores)]
            
            # 分配唯一对象ID (从1开始)
            obj_id = obj_idx + 1
            combined_mask[best_mask] = obj_id
            object_ids.append(obj_id)
        
        # 初始化CUTIE处理器
        mask_tensor = torch.from_numpy(combined_mask).to(self.device)
        self.processor.clear_memory()
        self.processor.step(to_tensor(rgb_frame).to(self.device), mask_tensor, object_ids)
        
        # 更新状态
        self.current_objects = object_ids
        self.is_initialized = True
        
        return combined_mask

    def update_masks(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        更新多目标跟踪结果
        
        参数:
        frame: RGB格式的新帧 [H, W, 3]
        
        返回:
        list: 每个目标的二值mask列表 [mask1, mask2, ...]
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_masks first.")
        
        # 准备输入数据
        rgb_frame = frame
        image_tensor = to_tensor(rgb_frame).to(self.device)
        
        # CUTIE推理
        with torch.no_grad():
            output_prob = self.processor.step(image_tensor)
            current_mask = self.processor.output_prob_to_mask(output_prob)
            current_mask_np = current_mask.cpu().numpy().astype(np.uint8)
        
        # 分离各个目标的mask
        return current_mask_np,[(current_mask_np == obj_id) for obj_id in self.current_objects]

    def reset(self):
        """重置管道状态"""
        self.processor.clear_memory()
        self.current_objects = []
        self.is_initialized = False

    def add_object(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        动态添加新目标到现有跟踪
        
        参数:
        frame: RGB格式的当前帧
        bbox: 新目标的边界框 [x1, y1, x2, y2]
        
        返回:
        new_mask: 新目标的单独mask
        """
        # 生成新目标mask
        rgb_frame = frame[..., ::-1]
        self.predictor.set_image(rgb_frame)
        masks, scores, _ = self.predictor.predict(box=np.array(bbox), multimask_output=True)
        new_mask = masks[np.argmax(scores)]
        
        # 分配新ID
        new_id = max(self.current_objects) + 1 if self.current_objects else 1
        new_mask_tensor = torch.from_numpy(new_mask.astype(np.uint8) * new_id).to(self.device)
        
        # 合并到现有mask
        combined_mask = self.processor.output_prob_to_mask(self.processor.prob)
        combined_mask = torch.where(new_mask_tensor > 0, new_mask_tensor, combined_mask)
        
        # 更新处理器状态
        self.current_objects.append(new_id)
        self.processor.step(image_tensor, combined_mask, self.current_objects)
        
        return new_mask

class RealsenseCamera:
    # TODO havent been tested
    def __init__(self):
        self.pipeline = rs.pipeline()  # type: ignore
        config = rs.config()  # type: ignore
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # type: ignore
        
    def __enter__(self):
        self.pipeline.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipeline.stop()
        
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        return frame

def main_rs(frame_callback:Callable|None):
    # 初始化所有模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    vl_adapter = QwenVLAdapter(model_path="/data/model/Qwen/Qwen2.5-VL-3B-Instruct")
    
    with torch.no_grad(), RealsenseCamera() as camera:
        cutie_model = get_default_model()
        pipeline = ImagePipeline(sam_checkpoint, sam_model_type, cutie_model, vl_adapter)
        
        # 初始化标志
        initialized = False
        
        while True:
            frame = camera.get_frame()
            rgb_frame = frame[:,:,::-1].copy()  # BGR转RGB
            
            if not initialized:
                # 使用第一帧进行初始化
                combined_mask, bboxes = pipeline.initialize_with_instruction(
                    frame=frame,
                    instruction="TRASH",
                    return_bbox=True
                )
                initialized = True
                centers = compute_mask_centers(combined_mask, 'centroid')
            else:
                # 更新掩码
                updated_mask, _ = pipeline.update_masks(rgb_frame)
                centers = compute_mask_centers(updated_mask, 'centroid')
                
            # 实时可视化
            vis_frame = frame.copy()
            visualize(vis_frame, bboxes=bboxes if not initialized else None, mask=updated_mask)
            visualize_centers(vis_frame, centers=centers)
            frame_callback(centers) if frame_callback else None
            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()    
    
def main_demo():
    # 初始化所有模型
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    vl_adapter = QwenVLAdapter(model_path="/data/model/Qwen/Qwen2.5-VL-3B-Instruct")
    with torch.no_grad():
        cutie_model = get_default_model()
        # 创建增强版管道
        pipeline = ImagePipeline(sam_checkpoint, sam_model_type, cutie_model, vl_adapter)
        
        # 通过自然语言指令初始化
        frame = cv2.imread("erqa.png")
        rgb_frame=  frame[:,:,::-1].copy()
        combined_mask, bboxes = pipeline.initialize_with_instruction(
            frame=frame,
            instruction="TRASH",  # 自然语言指令
            return_bbox=True
        )
        
        # 可视化结果
        visualize(frame,bboxes=bboxes, mask=combined_mask)
        centers = compute_mask_centers(combined_mask,'centroid')
        visualize_centers(frame, centers=centers)
        # 后续跟踪流程...
        for _ in range(10):
            new_frame = rgb_frame  # 获取新帧
            updated_mask, _ = pipeline.update_masks(new_frame)
            centers = compute_mask_centers(updated_mask,'centroid')
            visualize(new_frame, mask=updated_mask)
            visualize_centers(new_frame, centers=centers)
    
# 使用示例
if __name__ == "__main__":
    control_fn = None # 每个相机帧的控制回调函数，输入为list[tuple[float,float]] -> [(x_1,y_1), (x_2,y_2)] 如果有多个mask则对应多个 x,y
    main_rs(control_fn)