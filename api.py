from __future__ import annotations

import importlib
import sys

MODULE_NAME = "sentinel_pro.api.app"

if MODULE_NAME in sys.modules:
    _app_module = importlib.reload(sys.modules[MODULE_NAME])
else:
    _app_module = importlib.import_module(MODULE_NAME)

app = _app_module.app


def __getattr__(name: str):
    return getattr(_app_module, name)


def __dir__():
    return sorted({*globals(), *dir(_app_module)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
