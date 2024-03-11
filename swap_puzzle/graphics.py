import pygame
import pygame.freetype
from grid import Grid

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
GREY = pygame.Color(100, 100, 100)
AZUR_BRUME = pygame.Color("#faf0e6")
RED = pygame.Color("#b90e0a")
BLUE = pygame.Color("#4169e1")
GREEN = pygame.Color("#4cbb17")


def aa_filled_rounded_rect(
    surface: pygame.Surface, rect: pygame.Rect, color: pygame.Color, radius: float
) -> pygame.Rect:
    """
    Draw an antialiased and filled rectangle with rounded corners.
    Taken from https://www.pygame.org/project-AAfilledRoundedRect-2349-.html

    Parameters:
    -----------
    surface: pygame.Surface
        The surface of destination
    rect: pygame.Rectangle
        The rectangle to draw.
    color: pygame.Color
        The color to fill in the rectangle.
    radius: float
        The ratio of the radius of the circle with the shortest side of the rectangle.
        Must be between 0 and 1.

    Output:
    -------
    rect: pygame.rectangle
        The area of the affected pixels, excluding any pixels outside the destination Surface,
        or outside the clipping area.
    """

    rect = pygame.Rect(rect)
    color = pygame.Color(*color)
    alpha = color.a
    color.a = 0
    pos = rect.topleft
    rect.topleft = 0, 0
    rect2 = pygame.Surface(rect.size, pygame.SRCALPHA)

    circle = pygame.Surface([min(rect.size) * 3] * 2, pygame.SRCALPHA)
    pygame.draw.ellipse(circle, (0, 0, 0), circle.get_rect(), 0)
    circle = pygame.transform.smoothscale(circle, [int(min(rect.size) * radius)] * 2)

    rect3 = rect2.blit(circle, (0, 0))
    rect3.bottomright = rect.bottomright
    rect2.blit(circle, rect3)
    rect3.topright = rect.topright
    rect2.blit(circle, rect3)
    rect3.bottomleft = rect.bottomleft
    rect2.blit(circle, rect3)

    rect2.fill((0, 0, 0), rect.inflate(-rect3.w, 0))
    rect2.fill((0, 0, 0), rect.inflate(0, -rect3.h))

    rect2.fill(color, special_flags=pygame.BLEND_RGBA_MAX)
    rect2.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MIN)

    return surface.blit(rect2, pos)


def game(grid: Grid, W: int, H: int) -> None:
    """
    Launches a graphical user interface for playing the swap puzzle.

    Parameters:
    -----------
    grid: Grid
        The grid to be played.
    W, H: int
        The width and height of the window.
    """
    pygame.init()
    pygame.display.set_caption("Swap Puzzle")
    clock = pygame.time.Clock()
    clock.tick(60)
    square_padding = W // 100
    square_size = (W - square_padding) // grid.m
    font_size = square_size // 2
    offset = square_size // 25

    font = pygame.freetype.SysFont("Arial", font_size, bold=True)
    screen = pygame.display.set_mode((W, H))
    running = True
    clicked = (-1, -1)  # Last cell clicked
    past_clicked = (-1, -1)  # Second to last cell clicked
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                past_clicked = clicked
                (x, y) = pygame.mouse.get_pos()
                j = (x - square_padding) // square_size
                i = (y - square_padding) // square_size
                if clicked == (i, j):
                    clicked = (-1, -1)
                else:
                    clicked = (i, j)
        if past_clicked != (-1, -1) and clicked != (-1, -1):
            if clicked in grid.allowed_swaps(past_clicked):
                grid.swap(past_clicked, clicked)
            clicked = (-1, -1)
            past_clicked = (-1, -1)
        screen.fill(AZUR_BRUME)
        for i in range(grid.m):
            for j in range(grid.n):
                color = RED
                if grid.state[i][j] == i * grid.n + j + 1:
                    color = GREEN
                if clicked == (i, j):
                    color = BLUE
                rect = pygame.Rect(
                    j * square_size + square_padding,
                    i * square_size + square_padding,
                    square_size - square_padding,
                    square_size - square_padding,
                )
                aa_filled_rounded_rect(screen, rect, color, 0.3)

        for i in range(grid.m):
            for j in range(grid.n):
                text_surface, rect = font.render(str(grid.state[i][j]), GREY)
                rect.centerx = (
                    j * square_size
                    + square_padding
                    + (square_size - square_padding) // 2
                    + offset
                )
                rect.centery = (
                    i * square_size
                    + square_padding
                    + (square_size - square_padding) // 2
                    + offset
                )
                screen.blit(text_surface, rect)

                text_surface, rect = font.render(str(grid.state[i][j]), WHITE)
                rect.centerx = (
                    j * square_size
                    + square_padding
                    + (square_size - square_padding) // 2
                )
                rect.centery = (
                    i * square_size
                    + square_padding
                    + (square_size - square_padding) // 2
                )
                screen.blit(text_surface, rect)

        pygame.display.flip()
    pygame.quit()