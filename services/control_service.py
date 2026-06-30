"""Control service: manages display params and ZED camera lifecycle."""

from models import DisplayParamsStore, current_params_store, DEFAULT_DISPLAY_PARAMS
from .persistence import load_params, save_params


class ControlService:
    def __init__(self, params_store: DisplayParamsStore) -> None:
        self._params = params_store
        self._load()

    def _load(self) -> None:
        saved = load_params()
        if saved:
            self._params.update(saved)

    def get_params(self) -> dict:
        return self._params.get()

    def update_params(self, params: dict) -> dict:
        self._params.update(params)
        result = self._params.get()
        save_params(result)
        return result

    def reset_params(self) -> dict:
        self._params.reset()
        save_params(self._params.get())
        return self._params.get()


control_service = ControlService(current_params_store)
