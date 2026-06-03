from importlib import import_module

__all__ = ["QwenMDPOTrainer"]


def __getattr__(name):
    if name == "QwenMDPOTrainer":
        module = import_module(".mdpo_trainer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
