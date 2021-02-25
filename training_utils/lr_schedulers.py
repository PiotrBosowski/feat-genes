import torch


class CyclicLRScheduler:
    """Cyclic Learning Rate Scheduler spawner."""

    def __init__(self, min_lr_divider, step_size_up):
        """
        Instantiates the scheduler spawner.

        :param min_lr_divider: used to calculate minimum learning rate
        as self.max_lr / min_lr_divider
        :param step_size_up: number of steps of increasing learning rate
        """
        self.min_lr_divider = min_lr_divider
        self.step_size_up = step_size_up

    def __str__(self):
        return f"CyclicLR(minlr:{self.min_lr_divider}, stepup:{self.step_size_up})"

    def __call__(self, optimizer, max_lr):
        """
        Instantiates the CyclicLR scheduler itself.

        :param optimizer: optimizer the scheduler will operate on
        :param max_lr: max_lr the scheduler will reach
        :return: spawned CyclicLR scheduler
        """
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=max_lr / self.min_lr_divider, max_lr=max_lr,
            step_size_up=self.step_size_up, cycle_momentum=False)
        # cycle_momentum=True doesnt work with ADAM optimizer
