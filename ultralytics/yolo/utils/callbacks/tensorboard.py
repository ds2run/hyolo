# Ultralytics YOLO ??, GPL-3.0 license
from ultralytics.yolo.utils import LOGGER, TESTS_RUNNING, colorstr, SETTINGS
import contextlib
import warnings
from copy import deepcopy
from ultralytics.yolo.utils.torch_utils import de_parallel, torch

try:
    from torch.utils.tensorboard import SummaryWriter
    assert not TESTS_RUNNING  # do not log pytest
except (ImportError, AssertionError):
    SummaryWriter = None

writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    if writer:
        for k, v in scalars.items():
            writer.add_scalar(k, v, step)


def _log_gradients(gradient, step=0):
    if writer:
        writer.add_scalar('gradient', gradient, step)


def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard."""

    # Input image
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())  # for device, type
    im = torch.zeros((1, 3, *imgsz), device=p.device,
                     dtype=p.dtype)  # input image (must be zeros, not empty)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=UserWarning)  # suppress jit trace warning
        warnings.simplefilter(
            "ignore",
            category=torch.jit.TracerWarning)  # suppress jit trace warning

        # Try simple method first (YOLO)
        with contextlib.suppress(Exception):
            trainer.model.eval(
            )  # place in .eval() mode to avoid BatchNorm statistics changes
            writer.add_graph(
                torch.jit.trace(de_parallel(trainer.model), im, strict=False),
                [])
            LOGGER.info(
                f"{colorstr('TensorBoard: ')} model graph visualization added ?"
            )
            return

        # Fallback to TorchScript export steps (RTDETR)
        try:
            model = deepcopy(de_parallel(trainer.model))
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(
                        m, "export"
                ):  # Detect, RTDETRDecoder (Segment and Pose use Detect base class)
                    m.export = True
                    m.format = "torchscript"
            model(im)  # dry run

            writer.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(
                f"{colorstr('TensorBoard: ')} model graph visualization added ?"
            )
        except Exception as e:
            LOGGER.warning(
                f"{colorstr('TensorBoard: ')}WARNING ?? TensorBoard graph visualization failure {e}"
            )


def on_pretrain_routine_start(trainer):
    if SummaryWriter:
        try:
            global writer
            writer = SummaryWriter(str(trainer.save_dir))
            prefix = colorstr('TensorBoard: ')
            LOGGER.info(
                f"{prefix}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/"
            )
        except Exception as e:
            LOGGER.warning(
                f'WARNING ?? TensorBoard not initialized correctly, not logging this run. {e}'
            )


def on_train_start(trainer):
    """Log TensorBoard graph."""
    if writer:
        _log_tensorboard_graph(trainer)


def on_batch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix='train'),
                 trainer.epoch + 1)


def on_fit_epoch_end(trainer):
    res = {}
    for key in trainer.metrics:
        for k, v in trainer.metrics[key].items():
            res[str(key) + "_" + str(k)] = v
    _log_scalars(res, trainer.epoch + 1)


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_batch_end': on_batch_end,
    "on_train_start": on_train_start
}
