def get_learning_rates(base, step, n):
    """
    Calculates subsequent elements of a geometric sequence of provided
    parameters.

    :param base: base of the sequence
    :param step: step of the sequence
    :param n: number of elements
    :return: list of n subsequent elements
    """
    for index in range(n):
        base *= step
        yield base / step
