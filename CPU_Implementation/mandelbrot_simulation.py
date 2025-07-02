import pygame
import numpy as np
from Complex import Complex
from numba import cuda
import numba as nb


MAX_ITER = 100
WIDTH, HEIGHT = 800, 600

def complex_add(a: Complex, b: Complex):
    return Complex(a.real+b.real, a.imag+b.imag)

def complex_sqr(a: Complex):

    real = a.real**2 - a.imag**2
    imag = 2 * a.real * a.imag
    return Complex(real, imag)

def complex_abs(a: Complex):
    #return ((a.real**2 + a.imag**2) ** 0.5)
    #To avoid a square root operation, we can compare the value by 4 instead of 2
    return ((a.real**2 + a.imag**2) )

def get_color(iter):
    if iter == MAX_ITER:
        return(0, 0, 0)
    
    t = iter/MAX_ITER
    greyscale = int(255*t)
    r,g,b = greyscale,greyscale,greyscale
    # r = int(255 * (0.5 + 0.5 * np.sin(3 * t)))
    # g = int(255 * (0.5 + 0.5 * np.sin(3 * t + 2)))
    # b = int(255 * (0.5 + 0.5 * np.sin(3 * t + 4)))
    
    return (r, g, b)


def mandel_val(c: Complex):
    z = Complex(0, 0)
    for i in range(MAX_ITER):
        #z = z**2 + c
        z = complex_add(c, complex_sqr(z))
        #4 instead of 2 to avoid the square root operation:
        if complex_abs(z) > 4:
            return i
    return MAX_ITER

def mandelbrot_set(xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, WIDTH)
    y = np.linspace(ymin, ymax, HEIGHT)

    mandel_set = np.zeros((WIDTH, HEIGHT), dtype = int)
    
    for i in range(len(x)):
        for j in range(len(y)):
            c = Complex(x[i], y[j])
            mandel_set[i, j] = mandel_val(c)
    return mandel_set

def draw_mandelbrot(mandel_set):
    surface = pygame.Surface((WIDTH, HEIGHT))
    pixels = pygame.PixelArray(surface)

    for i in range(WIDTH):
        for j in range(HEIGHT):
            pixels[i, j] = get_color(mandel_set[i, j])

    #pixels = mandel_set

    return surface


def start():

    pygame.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill("black")
    xmin, xmax = -2.5, 1.5
    ymin, ymax = -1.5, 1.5

    zoom_factor = 0.5

    mandelbrot = mandelbrot_set(xmin, xmax, ymin, ymax)

    surface = draw_mandelbrot(mandelbrot)

    running = True

    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()

                cx = xmin + (xmax - xmin) * (x / WIDTH)
                cy = ymin + (ymax - ymin) * (y / HEIGHT)

                dx = (xmax-xmin) * zoom_factor/2
                dy = (ymax-ymin) * zoom_factor/2
                
                xmin = cx - dx
                xmax = cx + dx
                ymin = cy - dy
                ymax = cy + dy

                mandelbrot = mandelbrot_set(xmin, xmax, ymin, ymax)
                surface = draw_mandelbrot(mandelbrot)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()
start()