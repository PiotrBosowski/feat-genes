import math


def get_validation_steps(epoch_len, valids_per_epoch=4):
    """
    Returns the list of ticks at which validation should take place.
    The ticks are equally distributed. Validation at step 0 is skipped.
    Example: steps_in_epoch=140, valids_per_epoch=8, output:
    [16, 34, 51, 69, 86, 104, 121, 139]

    :param epoch_len: epoch length measured in steps (1 step = 1 batch)
    :param valids_per_epoch: requested number of validations per epoch
    :return: list of steps' indices in which validation happen
    """
    if valids_per_epoch < 1:
        return []
    mean_step = float(epoch_len - 1) / valids_per_epoch
    return [math.floor((i + 1) * mean_step) for i in range(valids_per_epoch)]
