import uuid
import torch
import torch.nn as nn
from Miscellaneous.User import get_current_user


class MainModule(nn.Module):

    def __init__(self,
                 model_name: str | None = None,
                 trace_model: bool = False):

        """
        :param trace_model: Whether to trace layers of the model
        :param model_name: The name of the model
        """

        super(MainModule, self).__init__()
        self._model_name = model_name

        if self._model_name:
            self._model_name = f"{get_current_user()['username']}_MM-{str(uuid.uuid4())}"

        self._trace_model = trace_model

    def output_shape(self,
                     forward_tensor: torch.Tensor) -> tuple[int, ...]:
        """
        :param forward_tensor: A tensor of comparable shape to
        what will be fed through the model
        :return: The output shape
        :rtype: tuple of ints
        """
        return tuple(forward_tensor.size())
