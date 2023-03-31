from mindspore.train.callback._callbacks import Callback, _handle_loss

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

class WandbCallback(Callback):

    def __init__(self,
                 name: str,
                 project: str,
                 entity: str,
                 group: str,
                 dir: str,
                 config: str,
                 ):
        super().__init__()
        if wandb is None:
            raise ModuleNotFoundError(
                "No wandb installation found. Install wandb via pip."
            )
        self._name = name
        self._project_name = project
        self._group_name = group
        self.entity = entity
        self.dir = dir

        wandb.init(
            name=name,
            project=project,
            group=group,
            config=config,
            entity=entity,
            dir=dir
        )

        self._train_loss_history = []
        self._val_loss_history = []

    def on_train_epoch_begin(self, run_context):
        self._train_loss_history.clear()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        self._train_loss_history.append(loss)
        wandb.log({"train_loss": loss})

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        mean_loss = sum(self._train_loss_history) / len(self._train_loss_history)
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        wandb.log({"train_loss_epoch", mean_loss})

    def on_eval_epoch_begin(self, run_context):
        self._eval_loss_history.clear()

    def on_eval_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        self._eval_loss_history.append(loss)
        wandb.log({"val_loss": loss})

    def on_eval_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        mean_loss = sum(self._eval_loss_history) / len(self._eval_loss_history)
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        wandb.log({"val_loss_epoch", mean_loss})
    