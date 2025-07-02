import pygame
import numpy as np
import numba as nb
from numba import cuda

WIDTH, HEIGHT = 1200, 1000
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

    # Initial and target view bounds
    xmin, xmax = -2.5, 1.5
    ymin, ymax = -2.0, 2.0
    txmin, txmax = xmin, xmax
    tymin, tymax = ymin, ymax

    zoom_factor = 0.9
    move_speed = 0.1
    smoothing = 0.1  # How fast it interpolates toward target bounds

    def render(xmin, xmax, ymin, ymax):
        real_axis = np.linspace(xmin, xmax, WIDTH, dtype=np.float32)
        imag_axis = np.linspace(ymin, ymax, HEIGHT, dtype=np.float32)
        pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        d_real_axis = cuda.to_device(real_axis)
        d_imag_axis = cuda.to_device(imag_axis)
        d_pixels = cuda.to_device(pixels)

        threadsperblock = (16, 16)
        blockspergrid_x = (WIDTH + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (HEIGHT + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        mandelbrot_set_kernel[blockspergrid, threadsperblock](
            d_real_axis, d_imag_axis, d_pixels, MAX_ITER
        )

        pixels = d_pixels.copy_to_host()
        return pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))

    surface = render(xmin, xmax, ymin, ymax)

    clock = pygame.time.Clock()
    running = True
    while running:
        updated = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                dx = (txmax - txmin) * move_speed
                dy = (tymax - tymin) * move_speed

                if event.key == pygame.K_LEFT:
                    txmin -= dx
                    txmax -= dx
                elif event.key == pygame.K_RIGHT:
                    txmin += dx
                    txmax += dx
                elif event.key == pygame.K_UP:
                    tymin -= dy
                    tymax -= dy
                elif event.key == pygame.K_DOWN:
                    tymin += dy
                    tymax += dy
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    # Zoom in
                    xmid = (txmin + txmax) / 2
                    ymid = (tymin + tymax) / 2
                    xrange = (txmax - txmin) * zoom_factor
                    yrange = (tymax - tymin) * zoom_factor
                    txmin = xmid - xrange / 2
                    txmax = xmid + xrange / 2
                    tymin = ymid - yrange / 2
                    tymax = ymid + yrange / 2
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    # Zoom out
                    xmid = (txmin + txmax) / 2
                    ymid = (tymin + tymax) / 2
                    xrange = (txmax - txmin) / zoom_factor
                    yrange = (tymax - tymin) / zoom_factor
                    txmin = xmid - xrange / 2
                    txmax = xmid + xrange / 2
                    tymin = ymid - yrange / 2
                    tymax = ymid + yrange / 2

        # Interpolate toward target bounds
        dxmin = txmin - xmin
        dxmax = txmax - xmax
        dymin = tymin - ymin
        dymax = tymax - ymax

        if abs(dxmin) > 1e-9 or abs(dxmax) > 1e-9 or abs(dymin) > 1e-9 or abs(dymax) > 1e-9:
            xmin += dxmin * smoothing
            xmax += dxmax * smoothing
            ymin += dymin * smoothing
            ymax += dymax * smoothing
            surface = render(xmin, xmax, ymin, ymax)

        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)  # Limit FPS to make zooming visually smoother

    pygame.quit()
    cuda.close()


if __name__ == "__main__":
    start()