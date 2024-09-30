
import random


def random_color():
    """Generate a random tuple of (r, g, b)."""
    r = random.randint(0, 255)  # Random integer between 0 and 255 for red
    g = random.randint(0, 255)  # Random integer between 0 and 255 for green
    b = random.randint(0, 255)  # Random integer between 0 and 255 for blue
    return (r, g, b)
f = random_color()
print(f)