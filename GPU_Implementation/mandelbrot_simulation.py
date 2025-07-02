import pygame
import numpy as np
import numba as nb
from numba import cuda

WIDTH, HEIGHT = 1000, 800
MAX_ITER = 100


@cuda.jit(device=True)
def complex_add(a_real, a_imag, b_real, b_imag):
    real = a_real + b_real
    imag = a_imag + b_imag
    return real, imag

@cuda.jit(device=True)
def complex_sqr(a_real, a_imag):
    real = (a_real ** 2) - (a_imag ** 2)
    imag = 2.0 * a_real * a_imag
    return real, imag

@cuda.jit(device=True)
def complex_abs(a_real, a_imag):
    return (a_real ** 2) + (a_imag ** 2)

@cuda.jit(device=True)
def mandel_val(real, imag, max_iter):
    z_real, z_imag = 0.0, 0.0
    for i in range(max_iter):
        z_real, z_imag = complex_add(*complex_sqr(z_real, z_imag), real, imag)
        if complex_abs(z_real, z_imag) > 4.0:
            return i
    return max_iter

@cuda.jit(device=True)
def get_color(iters, max_iter):
    if iters == max_iter:
        return (0, 0, 0)
    t = iters/max_iter
    # greyscale = int(255*t)
    # r,g,b = greyscale,greyscale,greyscale
    r = int(9 * (1 - t) * t * t * t * 255)
    g = int(15 * (1 - t) * (1 - t) * t * t * 255)
    b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
    
    return (r, g, b)

@cuda.jit
def mandelbrot_set_kernel(d_real_axis, d_imag_axis, d_pixels, max_iter):

    x, y = cuda.grid(2)
    if x >= d_real_axis.size or y >= d_imag_axis.size:
        return
    
    real = d_real_axis[x]
    imag = d_imag_axis[y]
    
    # d_mandel_set[y, x] = mandel_val(real, imag, max_iter)
    iter = mandel_val(real, imag, max_iter)
    r, g, b = get_color(iter, max_iter)
    d_pixels[y, x, 0] = r
    d_pixels[y, x, 1] = g
    d_pixels[y, x, 2] = b

def start():

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandelbrot Set (GPU)")
    screen.fill("black")

    xmin, xmax = -2.5, 1.5
    ymin, ymax = -1.5, 1.5

    zoom_factor = 0.9

    real_axis = np.linspace(xmin, xmax, WIDTH, dtype=np.float64)
    imag_axis = np.linspace(ymin, ymax, HEIGHT, dtype=np.float64)

    # mandel_set = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


    d_real_axis = cuda.to_device(real_axis)
    d_imag_axis = cuda.to_device(imag_axis)
    # d_mandel_set = cuda.to_device(mandel_set)
    d_pixels = cuda.to_device(pixels)

    threadsperblock = (16, 16)
    blockspergird_x = (WIDTH + threadsperblock[0]-1)//threadsperblock[0]
    blockspergird_y = (HEIGHT + threadsperblock[1]-1 )//threadsperblock[1]
    blockspergrid = (blockspergird_x, blockspergird_y)

    mandelbrot_set_kernel[blockspergrid, threadsperblock](d_real_axis, d_imag_axis, d_pixels, MAX_ITER)

    pixels = d_pixels.copy_to_host()
    surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()

                cx = xmin + (xmax - xmin) * (x / WIDTH)
                cy = ymin + (ymax - ymin) * (y / HEIGHT)

                dx = (xmax - xmin) * zoom_factor/2
                dy = (ymax - ymin) * zoom_factor/2

                xmin = cx - dx
                xmax = cx + dx
                ymin = cy - dy
                ymax = cy + dy

                real_axis = np.linspace(xmin, xmax, WIDTH, dtype=np.float32)
                imag_axis = np.linspace(ymin, ymax, HEIGHT, dtype=np.float32)

                d_real_axis = cuda.to_device(real_axis)
                d_imag_axis = cuda.to_device(imag_axis)
                d_pixels = cuda.to_device(pixels)

                mandelbrot_set_kernel[blockspergrid, threadsperblock](d_real_axis, d_imag_axis, d_pixels, MAX_ITER)

                pixels = d_pixels.copy_to_host()
                surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))


        screen.blit(surface, (0, 0))
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    start()