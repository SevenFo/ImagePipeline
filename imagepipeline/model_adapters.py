from abc import ABC, abstractmethod
import torch
# from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor
# from qwen_omni_utils import process_mm_info
from qwen_vl_utils import process_vision_info
import time
# import soundfile
import openai
import numpy as np

import pdb

# from DexGraspVLA.planner.modeling_qwen2_5_omni_stream import Qwen2_5OmniForConditionalGenerationStreaming
# from DexGraspVLA.inference_utils.audio_player import AudioPlayer
class BaseModelAdapter(ABC):
    @abstractmethod
    def prepare_input(self, text: str, audio_path: str = None, image_url: str = None) -> list[dict]:
        pass

    @abstractmethod
    def generate_response(self, input_data:list[dict]):
        """generate response

        Args:
            input_data (list[dict]): processed data returned by `self.prepare_input`

        Returns:
            tuple[std,std]: [generated text, generated audio path (None means no audio generated)]
        """
        pass

class QwenOmniAdapter(BaseModelAdapter):
    def __init__(self, model_path="/data/model/Qwen/Qwen2.5-Omni-7B"):
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    def prepare_input(self, text: str, audio_path: str = None, image_url: str = None) -> list[dict]:
        messages = [
            {"role": "system",
             "content":
                 [
                     {
                         "type":"text",
                         "text":'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
                         }
                     ]
                 },
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        if audio_path:
            messages[1]["content"].append({"type": "audio", "audio": "/data/shiqi/SRCB-DexGraspVLA-Project/user_prompt.wav"})
            messages[1]["content"].append({"type": "audio", "audio": audio_path})
        if image_url:
            # messages[1]["content"].append({"type": "text", "text": "following is an image"})
            messages[1]["content"].append({"type": "image", "image": image_url})
        # messages[1]["content"].append({"type": "text", "text": "\n\n The text you generated must follow the 'Expected output format' in json, while speech is for communication"})
        return messages

    def generate_response(self, input_data, max_tokens, **kwargs):
        messages = input_data
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        output = self.model.generate(**inputs, return_audio=True, use_audio_in_video=True, max_new_tokens = max_tokens, **kwargs)
        if output[1] is not None:
            print(f"with audio {output[1]}")
            audio = soundfile.write(f"{time.time_ns() // 1000000}.wav",output[1].detach().cpu().numpy(),samplerate=24000)
            return self.processor.batch_decode(output[0], skip_special_tokens=True)[0], output[1]
        else:
            return self.processor.batch_decode(output, skip_special_tokens=True)[0], None

class QwenOmniStreamAdapter(QwenOmniAdapter):
    def __init__(self, model_path="/data/model/Qwen/Qwen2.5-Omni-7B"):
        self.model = Qwen2_5OmniForConditionalGenerationStreaming.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        self.audio_player = AudioPlayer()

    def generate_response(self, input_data:list[dict], max_tokens, **kwargs):
        return_audio=kwargs.pop("return_audio", False)     
        audio_device_name = kwargs.pop("audio_device_name","default")   
        messages = input_data
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        output_steam = self.model.generate_stream(inputs = inputs, return_audio=return_audio, use_audio_in_video=True, thinker_max_new_tokens = max_tokens, thinker_temperature = 0.3, **kwargs)
        
        audio_output = []
        text_output = []
        
        self.audio_player.configure_stream(rate = 24000,channels=1,sampwidth=4, audio_device_name= audio_device_name)
        
        try:
            for output_chunk in output_steam:
                # if output_chunk[1] is not None:
                if return_audio:
                    text_chunk = output_chunk['thinker_ids']
                    audio_chunk:np.ndarray = output_chunk['talker_wav'].detach().cpu().numpy()
                    print(f"with audio {len(audio_chunk)}")
                    audio_output.append(audio_chunk)
                    self.audio_player.add_chunk(audio_chunk.astype(np.float32))
                    if (output_chunk['is_new_thinker_ids']):
                        text_chunk = self.processor.batch_decode(text_chunk, skip_special_tokens=True)[0]
                        text_output.append(text_chunk)
                        print(text_chunk)
                else:
                    text_chunk = output_chunk['thinker_ids']
                    text_chunk = self.processor.batch_decode(text_chunk, skip_special_tokens=True)[0]
                    text_output.append(text_chunk)
                    print(text_chunk)

            self.audio_player.stop_current_stream_after_all_audio()
            
            audio = np.concatenate(audio_output) if audio_output else None
            text = "".join(text_output)
            soundfile.write(f"{time.time_ns() // 1000000}.wav",audio,samplerate=24000) if audio is not None else None
            return text, audio 
        except Exception as e:
            import traceback
            print(f"{type(e).__name__}: {e}")
            traceback.print_exc()  
            pdb.set_trace()  

class QwenVLAdapter(BaseModelAdapter):
    def __init__(self, model_path):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def prepare_input(self, text: str, audio_path: str = None, image_url: str = None) -> list[dict]:
        """构建Qwen-VL的输入格式"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]
        
        # 添加图像输入
        if image_url:
            messages[1]["content"].append({
                "type": "image",
                "image": image_url  # 假设image_url是base64编码或本地路径
            })
            
        return messages

    def generate_response(self, input_data, max_tokens, **kwargs):
        """生成文本响应"""
        messages = input_data
        
        # 处理多模态输入
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 分离视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 准备模型输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 生成响应
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **kwargs
        )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response, None  # Qwen-VL不支持音频输出
    
class OpenAIAdapter(BaseModelAdapter):
    def __init__(self, api_key: str, base_url: str = None, model_name: str = None):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = self.client.models.list().data[0].id if model_name is None else model_name

    def prepare_input(self, text: str, audio_path: str = None, image_url: str = None):
        # OpenAI 不支持音频输入，暂时忽略
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        ]
        if image_url is not None:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        return messages

    def generate_response(self, input_data, max_tokens, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=input_data,
            max_completion_tokens=max_tokens
            **kwargs
        )
        return response.choices[0].message.content, None
