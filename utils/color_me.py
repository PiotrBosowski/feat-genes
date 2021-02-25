class ColorMe:
    """
    Simple utility to easily color a text printed in the console.

    Usage: print(ColorMe("green")("Hello World!"))
    """

    colors = {
        'unchanged': "{0}",
        'yellow': "\033[93m{0}\033[00m",
        'sea': "\033[96m{0}\033[00m",
        'red': "\033[91m{0}\033[00m",
        'green': "\033[92m{0}\033[00m",
        'blue': "\033[34m{0}\033[00m",
        'blue_bg': "\033[44m{0}\033[00m",
        'purple_bg': "\033[45m{0}\033[00m",
        'red_bg': "\033[41m{0}\033[00m",
        'yellow_bg': "\033[43m{0}\033[00m",
        'green_bg': "\033[42m{0}\033[00m"
    }

    def __init__(self, color='unchanged'):
        color = color if color in ColorMe.colors else 'unchanged'
        self.color = ColorMe.colors[color]

    def __call__(self, text):
        return self.color.format(text)
