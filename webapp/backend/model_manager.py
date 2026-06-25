# model_manager.py
# =========================
# 模型管理器：负责加载 / 切换 YOLO 模型
# =========================

import os
from ultralytics import YOLO


# 每个检测模式的默认模型
MODE_DEFAULTS = {
    "image":  "best-v11m.pt",
    "video":  "best-v8n.pt",
    "camera": "best-v8n.pt",
}

VALID_MODES = set(MODE_DEFAULTS.keys())


class ModelManager:
    """
    YOLO 模型管理器
    支持：
    - 按模式（image / video / camera）独立加载 / 切换模型
    - 获取指定模式的模型
    - 列出可用模型文件
    """

    def __init__(self):
        # 每个 mode 独立存一个 (model_obj, model_name)
        self._models: dict = {}          # mode -> YOLO 实例
        self._model_names: dict = {}     # mode -> 模型文件名

    def _load(self, model_name: str):
        """内部：加载模型文件，返回 YOLO 实例"""
        model_path = os.path.join(
            os.path.dirname(__file__), "models", model_name
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        model = YOLO(model_path)
        model.model.names = {0: "smoke", 1: "fire"}   # 🔥 修正类别显示
        print(f"✅ Loaded model: {model_name}")
        return model

    def load_model(self, model_name: str, mode: str = "image"):
        """
        为指定模式加载模型
        :param model_name: models 目录下的文件名
        :param mode: 'image' | 'video' | 'camera'
        """
        if mode not in VALID_MODES:
            raise ValueError(f"❌ Unknown mode: {mode}. Use one of {VALID_MODES}")
        self._models[mode] = self._load(model_name)
        self._model_names[mode] = model_name

    def get_model(self, mode: str = "image"):
        """
        获取指定模式的模型；若该模式尚未加载则按默认值自动加载
        """
        if mode not in VALID_MODES:
            raise ValueError(f"❌ Unknown mode: {mode}")
        if mode not in self._models:
            default = MODE_DEFAULTS[mode]
            print(f"⚠️  Mode '{mode}' not loaded yet, loading default: {default}")
            self.load_model(default, mode)
        return self._models[mode]

    def switch_model(self, model_name: str, mode: str = "image"):
        """切换指定模式的模型"""
        if mode not in VALID_MODES:
            raise ValueError(f"❌ Unknown mode: {mode}")
        if self._model_names.get(mode) == model_name:
            print(f"⚠️ Mode '{mode}' already using {model_name}")
            return
        self.load_model(model_name, mode)
        print(f"🔄 Mode '{mode}' switched to: {model_name}")

    def current_name(self, mode: str = "image"):
        """返回指定模式当前加载的模型文件名"""
        return self._model_names.get(mode)

    def list_models(self) -> list:
        """列出 models 目录下所有可用的 .pt 文件"""
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(models_dir):
            return []
        return sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])


# 🔥 全局单例（整个系统只用这一个）
model_manager = ModelManager()