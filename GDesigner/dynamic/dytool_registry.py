from class_registry import ClassRegistry
from typing import Type
from GDesigner.dynamic.dytool import Dytool


class ToolRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, name: str, *args, **kwargs) -> Dytool:
        return cls.registry.get(name, *args, **kwargs)

    @classmethod
    def get_class(cls, name: str) -> Type:
        return cls.registry.get_class(name)