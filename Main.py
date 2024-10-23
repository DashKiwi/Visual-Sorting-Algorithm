import pygame
import numpy as np
import random
import math
pygame.init()

class DrawInfomation:
    LIST_COLOR = [(0, 128, 0), 
                  (0, 160, 0),
                  (0, 192, 0)]
    REPLACE_IN_LIST = 0, 0, 255
    CURRENT_LIST_POS = 255, 0, 0
    TITLE_TXT_COLOR = 255, 0, 0
    MINI_TXT_COLOR = 255, 255, 255
    BACKGROUND_COLOR = 0, 0, 0

    FONT = pygame.font.SysFont('comic', 30)
    LARGE_FONT = pygame.font.SysFont('comic', 40)
    TOP_PAD = 150
    SIDE_PAD = 100

    def __init__(self, width, height, lst):
        self.width = width
        self.height = height

        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Sorting Algorithm Visulation")
        self.set_list(lst)
    
    def set_list(self, lst):
        self.lst = lst
        self.max_val = max(lst)
        self.min_val = min(lst)

        self.block_width = math.floor((self.width - self.SIDE_PAD) / len(lst))
        self.block_height = math.floor((self.height - self.TOP_PAD) / (self.max_val - self.min_val))
        self.start_x = self.SIDE_PAD // 2

def draw(draw_info, algo_name, ascending):
    # Create the background
    draw_info.window.fill(draw_info.BACKGROUND_COLOR)
    title = draw_info.LARGE_FONT.render(f"{algo_name} - {'Ascending' if ascending else 'Descending'}", 1, draw_info.TITLE_TXT_COLOR)
    draw_info.window.blit(title, (draw_info.width / 2 - title.get_width() / 2, 5))

    controls = draw_info.FONT.render("R - Reset | SPACE - Start Sorting | A - Ascending | D - Descending", 1, draw_info.MINI_TXT_COLOR)
    draw_info.window.blit(controls, (draw_info.width / 2 - controls.get_width() / 2, 45))

    sorting = draw_info.FONT.render("S - Switch Algorithm", 1, draw_info.MINI_TXT_COLOR)
    draw_info.window.blit(sorting, (draw_info.width / 2 - sorting.get_width() / 2, 75))

    draw_list(draw_info)
    pygame.display.update()

def draw_list(draw_info, color_positions = {}, clear_bg = False):
    lst = draw_info.lst

    if clear_bg:
        clear_rect = (draw_info.SIDE_PAD // 2, draw_info.TOP_PAD, 
                      draw_info.width - draw_info.SIDE_PAD, 
                      draw_info.height - draw_info.TOP_PAD)
        pygame.draw.rect(draw_info.window, draw_info.BACKGROUND_COLOR, clear_rect)

    for i, val in enumerate(lst):
        x = draw_info.start_x + i * draw_info.block_width
        y = draw_info.height - (val - draw_info.min_val) * draw_info.block_height

        color = draw_info.LIST_COLOR[i % 3]
        if i in color_positions:
            color = color_positions[i]
        
        pygame.draw.rect(draw_info.window, color, (x, y, draw_info.block_width, draw_info.height))

    if clear_bg:
        pygame.display.update()

def generate_sine_wave(freq, duration=0.1, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)  # Mono wave
    wave_stereo = np.column_stack((wave, wave))  # Duplicate for stereo
    sound = pygame.sndarray.make_sound((wave_stereo * 32767).astype(np.int16))
    return sound
        
def generate_starting_list(n, min_val, max_val):
    lst = []

    for _ in range(n):
        val = random.randint(min_val, max_val)
        lst.append(val)
    
    return lst

def bubble_sort(draw_info, ascending=True):
    lst = draw_info.lst
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            num1 = lst[j]
            num2 = lst[j + 1]

            freq1 = map_value_to_freq(num1)
            freq2 = map_value_to_freq(num2)
            generate_sine_wave(freq1).play()
            generate_sine_wave(freq2).play()
            
            if (num1 > num2 and ascending) or (num1 < num2 and not ascending):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swap_freq = map_value_to_freq(num1)
                generate_sine_wave(swap_freq).play()
                draw_list(draw_info, {j: draw_info.CURRENT_LIST_POS, j + 1: draw_info.REPLACE_IN_LIST}, True)
                yield True
    return lst

def map_value_to_freq(value):
    # Map the value (1 to 100) to frequency (200 Hz to 1000 Hz)
    min_freq = 200
    max_freq = 1000
    return min_freq + (max_freq - min_freq) * (value - 1) / 99

def insertion_sort(draw_info, ascending=True):
    lst = draw_info.lst

    for i in range(1, len(lst)):
        current = lst[i]

        while True:
            ascending_sort = i > 0 and lst[i - 1] > current and ascending
            descending_sort = i > 0 and lst[i - 1] < current and not ascending

            if not ascending_sort and not descending_sort:
                break
            
            lst[i] = lst[i - 1]
            i = i - 1
            lst[i] = current
            draw_list(draw_info, {i - 1: draw_info.CURRENT_LIST_POS, i: draw_info.REPLACE_IN_LIST}, True)
            yield True
    
    return lst

def selection_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)
    
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if lst[j] < lst[min_idx] and ascending or lst[j] > lst[min_idx] and not ascending:
                min_idx = j
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, j: draw_info.REPLACE_IN_LIST}, True)
            yield True
                
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
        
    return lst

def counting_sort(draw_info, ascending=True):
    lst = draw_info.lst
    M = max(lst)

    count_array = [0] * (M + 1)

    for num in lst:
        count_array[num] += 1

    if ascending:
        for i in range(1, M + 1):
            count_array[i] += count_array[i - 1]
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS}, True)
            yield True

        output_array = [0] * len(lst)
        for i in range(len(lst) - 1, -1, -1):
            output_array[count_array[lst[i]] - 1] = lst[i]
            count_array[lst[i]] -= 1
            draw_info.lst = output_array
            draw_list(draw_info, {output_array[count_array[lst[i]] - 1]: draw_info.REPLACE_IN_LIST}, True)
            yield True
    else:
        for i in range(M - 1, -1, -1):
            count_array[i] += count_array[i + 1]
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS}, True)
            yield True

        output_array = [0] * len(lst)
        for i in range(len(lst)):
            output_array[count_array[lst[i]] - 1] = lst[i]
            count_array[lst[i]] -= 1
            draw_info.lst = output_array
            draw_list(draw_info, {output_array[count_array[lst[i]] - 1]: draw_info.REPLACE_IN_LIST}, True)
            yield True

    return output_array

def merge(draw_info, left, mid, right, ascending=True):
    left_half = draw_info.lst[left:mid + 1]
    right_half = draw_info.lst[mid + 1:right + 1]

    left_index, right_index = 0, 0
    sorted_index = left

    while left_index < len(left_half) and right_index < len(right_half):
        if (left_half[left_index] <= right_half[right_index] and ascending) or \
           (left_half[left_index] > right_half[right_index] and not ascending):
            draw_info.lst[sorted_index] = left_half[left_index]
            left_index += 1
        else:
            draw_info.lst[sorted_index] = right_half[right_index]
            right_index += 1

        draw_list(draw_info, {sorted_index: draw_info.REPLACE_IN_LIST}, True)
        yield True
        sorted_index += 1

    while left_index < len(left_half):
        draw_info.lst[sorted_index] = left_half[left_index]
        left_index += 1
        sorted_index += 1
        draw_list(draw_info, {sorted_index - 1: draw_info.REPLACE_IN_LIST}, True)
        yield True

    while right_index < len(right_half):
        draw_info.lst[sorted_index] = right_half[right_index]
        right_index += 1
        sorted_index += 1
        draw_list(draw_info, {sorted_index - 1: draw_info.REPLACE_IN_LIST}, True)
        yield True

def merge_sort(draw_info, left=0, right=None, ascending=True):
    if right is None:
        right = len(draw_info.lst) - 1

    if left < right:
        mid = (left + right) // 2
        
        yield from merge_sort(draw_info, left, mid, ascending)
        yield from merge_sort(draw_info, mid + 1, right, ascending)
        yield from merge(draw_info, left, mid, right, ascending)

def generate_1_to_100_list():
    lst = []

    for i in range(99):
        lst.append(i + 1)
    
    random.shuffle(lst)
    return lst

def main():
    run = True
    clock = pygame.time.Clock()

    n = 50
    min_val = 0
    max_val = 100

    #lst = generate_starting_list(n, min_val, max_val)
    lst = generate_1_to_100_list()
    draw_info = DrawInfomation(800, 600, lst)
    sorting = False
    ascending = True

    sorting_algorithm = bubble_sort
    sorting_algo_name = "Bubble Sort"
    sorting_algorithm_generator = None


    while run:
        clock.tick(60)

        if sorting:
            try:
                next(sorting_algorithm_generator)
            except StopIteration:
                sorting = False
        else:
            draw(draw_info, sorting_algo_name, ascending)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type != pygame.KEYDOWN:
                continue
            
            if event.key == pygame.K_r:
                #lst = generate_starting_list(n, min_val, max_val)
                lst = generate_1_to_100_list()
                draw_info.set_list(lst)
                sorting = False
            elif event.key == pygame.K_SPACE and not sorting:
                sorting = True
                if sorting_algorithm == merge_sort:
                    sorting_algorithm_generator = sorting_algorithm(draw_info, 0, len(draw_info.lst) - 1, ascending)
                else:
                    sorting_algorithm_generator = sorting_algorithm(draw_info, ascending)
            elif event.key == pygame.K_a and not sorting:
                ascending = True
            elif event.key == pygame.K_d and not sorting:
                ascending = False
            elif event.key == pygame.K_s and not sorting:
                if sorting_algo_name == "Bubble Sort":
                    sorting_algorithm = insertion_sort
                    sorting_algo_name = "Insertion Sort"
                elif sorting_algo_name == "Insertion Sort":
                    sorting_algorithm = selection_sort
                    sorting_algo_name = "Selection Sort"
                elif sorting_algo_name == "Selection Sort":
                    sorting_algorithm = counting_sort
                    sorting_algo_name = "Counting Sort"
                elif sorting_algo_name == "Counting Sort":
                    sorting_algorithm = merge_sort
                    sorting_algo_name = "Merge Sort"
                elif sorting_algo_name == "Merge Sort":
                    sorting_algorithm = bubble_sort
                    sorting_algo_name = "Bubble Sort"

    
    pygame.quit()

if __name__ == "__main__":
    main()
