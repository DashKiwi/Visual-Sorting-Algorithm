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
    wave = 0.5 * np.sin(2 * np.pi * (freq + 150) * t)
    wave += 0.3 * np.sin(2 * np.pi * (freq * 2 + 150) * t)
    decay = np.exp(-5 * t)
    wave *= decay
    wave_stereo = np.column_stack((wave, wave))
    sound = pygame.sndarray.make_sound((wave_stereo * 32767).astype(np.int16))
    sound.set_volume(1.0)
    return sound

def generate_starting_list(n, min_val, max_val):
    lst = []

    for _ in range(n):
        val = random.randint(min_val, max_val)
        lst.append(val)
    
    return lst

def map_value_to_freq(value):
    min_freq = 400
    max_freq = 1200
    return min_freq + (max_freq - min_freq) * (value / 100)

def play_sound(freq):
    sound = generate_sine_wave(freq)
    sound.play()

def bubble_sort(draw_info, ascending=True):
    lst = draw_info.lst
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            num1 = lst[j]
            num2 = lst[j + 1]

            #freq1 = map_value_to_freq(num1)
            freq2 = map_value_to_freq(num2)
            #play_sound(freq1)
            play_sound(freq2)

            if (num1 > num2 and ascending) or (num1 < num2 and not ascending):
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                #swap_freq = map_value_to_freq(num1)
                #play_sound(swap_freq)
                draw_list(draw_info, {j: draw_info.CURRENT_LIST_POS, j + 1: draw_info.REPLACE_IN_LIST}, True)
                yield True
    return lst

def insertion_sort(draw_info, ascending=True):
    lst = draw_info.lst

    for i in range(1, len(lst)):
        current = lst[i]
        freq_current = map_value_to_freq(current)
        play_sound(freq_current)

        while True:
            ascending_sort = i > 0 and lst[i - 1] > current and ascending
            descending_sort = i > 0 and lst[i - 1] < current and not ascending

            if not ascending_sort and not descending_sort:
                break
            
            lst[i] = lst[i - 1]
            freq_swap = map_value_to_freq(lst[i - 1])
            play_sound(freq_swap)
            i = i - 1
            
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, i + 1: draw_info.REPLACE_IN_LIST}, True)
            yield True
            
        lst[i] = current
        draw_list(draw_info, {i: draw_info.REPLACE_IN_LIST}, True)
        yield True
    
    return lst

def selection_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)
    
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            freq_j = map_value_to_freq(lst[j])
            play_sound(freq_j)

            if lst[j] < lst[min_idx] and ascending or lst[j] > lst[min_idx] and not ascending:
                min_idx = j
            
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, j: draw_info.REPLACE_IN_LIST}, True)
            yield True
                
        lst[i], lst[min_idx] = lst[min_idx], lst[i]
        swap_freq = map_value_to_freq(lst[i])
        play_sound(swap_freq)
        
    return lst

def counting_sort(draw_info, ascending=True):
    lst = draw_info.lst
    M = max(lst)

    count_array = [0] * (M + 1)

    for num in lst:
        count_array[num] += 1
        freq_count = map_value_to_freq(num)
        play_sound(freq_count)
        draw_list(draw_info, {num: draw_info.CURRENT_LIST_POS}, True)
        yield True

    if ascending:
        for i in range(1, M + 1):
            count_array[i] += count_array[i - 1]
            freq_index = map_value_to_freq(i)
            play_sound(freq_index)
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS}, True)
            yield True

        output_array = [0] * len(lst)
        for i in range(len(lst) - 1, -1, -1):
            output_array[count_array[lst[i]] - 1] = lst[i]
            count_array[lst[i]] -= 1
            draw_info.lst = output_array
            freq_output = map_value_to_freq(output_array[count_array[lst[i]]])
            play_sound(freq_output)
            draw_list(draw_info, {output_array[count_array[lst[i]] - 1]: draw_info.REPLACE_IN_LIST}, True)
            yield True
    else:
        for i in range(M - 1, -1, -1):
            count_array[i] += count_array[i + 1]
            freq_index = map_value_to_freq(i)
            play_sound(freq_index)
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS}, True)
            yield True

        output_array = [0] * len(lst)
        for i in range(len(lst)):
            output_array[count_array[lst[i]] - 1] = lst[i]
            count_array[lst[i]] -= 1
            draw_info.lst = output_array
            freq_output = map_value_to_freq(output_array[count_array[lst[i]]])
            play_sound(freq_output)
            draw_list(draw_info, {output_array[count_array[lst[i]] - 1]: draw_info.REPLACE_IN_LIST}, True)
            yield True

    return output_array


def merge(draw_info, left, mid, right, ascending=True):
    left_half = draw_info.lst[left:mid + 1]
    right_half = draw_info.lst[mid + 1:right + 1]

    left_index, right_index = 0, 0
    sorted_index = left

    while left_index < len(left_half) and right_index < len(right_half):
        freq_left = map_value_to_freq(left_half[left_index])
        freq_right = map_value_to_freq(right_half[right_index])
        play_sound(freq_left)
        play_sound(freq_right)

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

def shell_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = lst[i]
            j = i
            freq_temp = map_value_to_freq(temp)
            play_sound(freq_temp)

            while j >= gap and ((lst[j - gap] > temp and ascending) or (lst[j - gap] < temp and not ascending)):
                lst[j] = lst[j - gap]
                freq_swap = map_value_to_freq(lst[j])
                play_sound(freq_swap)
                draw_list(draw_info, {j: draw_info.CURRENT_LIST_POS, j - gap: draw_info.REPLACE_IN_LIST}, True)
                yield True
                j -= gap

            lst[j] = temp
            draw_list(draw_info, {j: draw_info.REPLACE_IN_LIST}, True)
            yield True
        
        gap //= 2

def radix_sort(draw_info, ascending=True):
    lst = draw_info.lst
    max_val = max(lst)

    exp = 1
    while max_val // exp > 0:
        yield from counting_sort_radix(draw_info, exp, ascending)
        exp *= 10

def counting_sort_radix(draw_info, exp, ascending):
    lst = draw_info.lst
    n = len(lst)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = (lst[i] // exp) % 10
        count[index] += 1
        freq_count = map_value_to_freq(lst[i])
        play_sound(freq_count)
        draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS}, True)
        yield True
    if ascending:
        for i in range(1, 10):
            count[i] += count[i - 1]
    else:
        for i in range(8, -1, -1):
            count[i] += count[i + 1]
    for i in range(n - 1, -1, -1):
        index = (lst[i] // exp) % 10
        output[count[index] - 1] = lst[i]
        count[index] -= 1

        draw_info.lst = output
        freq_output = map_value_to_freq(output[count[index]])
        play_sound(freq_output)
        draw_list(draw_info, {count[index]: draw_info.REPLACE_IN_LIST}, True)
        yield True

    for i in range(n):
        lst[i] = output[i]

def comb_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)
    gap = n
    shrink = 1.3
    sorted = False

    while not sorted:
        gap = max(1, int(gap / shrink))
        sorted = True

        for i in range(n - gap):
            num1 = lst[i]
            num2 = lst[i + gap]

            freq_num1 = map_value_to_freq(num1)
            freq_num2 = map_value_to_freq(num2)
            play_sound(freq_num1)
            play_sound(freq_num2)

            if (num1 > num2 and ascending) or (num1 < num2 and not ascending):
                lst[i], lst[i + gap] = lst[i + gap], lst[i]
                sorted = False
                
                freq_swap = map_value_to_freq(lst[i])
                play_sound(freq_swap)

                draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, i + gap: draw_info.REPLACE_IN_LIST}, True)
                yield True

        yield True

    return lst

def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def bogo_sort(draw_info, ascending=True):
    lst = draw_info.lst
    while not is_sorted(lst):
        random.shuffle(lst)
        for index in range(len(lst)):
            freq = map_value_to_freq(lst[index])
            play_sound(freq)
            draw_list(draw_info, {index: draw_info.CURRENT_LIST_POS}, True)
    draw_info.lst = lst

def quick_sort(draw_info, low, high, ascending=True):
    if low < high:
        pivot_index = yield from partition(draw_info, low, high, ascending)
        yield from quick_sort(draw_info, low, pivot_index - 1, ascending)
        yield from quick_sort(draw_info, pivot_index + 1, high, ascending)

def partition(draw_info, low, high, ascending):
    pivot = draw_info.lst[high]
    pivot_freq = map_value_to_freq(pivot)
    play_sound(pivot_freq)
    i = low - 1

    for j in range(low, high):
        current_freq = map_value_to_freq(draw_info.lst[j])
        play_sound(current_freq)
        if (draw_info.lst[j] < pivot and ascending) or (draw_info.lst[j] > pivot and not ascending):
            i += 1
            draw_info.lst[i], draw_info.lst[j] = draw_info.lst[j], draw_info.lst[i]
            swap_freq = map_value_to_freq(draw_info.lst[i])
            play_sound(swap_freq)
            draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, j: draw_info.REPLACE_IN_LIST}, True)
            yield True

    draw_info.lst[i + 1], draw_info.lst[high] = draw_info.lst[high], draw_info.lst[i + 1]
    pivot_swap_freq = map_value_to_freq(draw_info.lst[i + 1])
    play_sound(pivot_swap_freq)
    draw_list(draw_info, {i + 1: draw_info.REPLACE_IN_LIST}, True)
    yield True

    return i + 1

def heapify(draw_info, n, i, ascending=True):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n:
        freq_left = map_value_to_freq(draw_info.lst[left])
        play_sound(freq_left)

        if (draw_info.lst[left] > draw_info.lst[largest] and ascending) or (draw_info.lst[left] < draw_info.lst[largest] and not ascending):
            largest = left

    if right < n:
        freq_right = map_value_to_freq(draw_info.lst[right])
        play_sound(freq_right)

        if (draw_info.lst[right] > draw_info.lst[largest] and ascending) or (draw_info.lst[right] < draw_info.lst[largest] and not ascending):
            largest = right

    if largest != i:
        draw_info.lst[i], draw_info.lst[largest] = draw_info.lst[largest], draw_info.lst[i]
        draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, largest: draw_info.REPLACE_IN_LIST}, True)
        yield True

        yield from heapify(draw_info, n, largest, ascending)

def heap_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)

    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(draw_info, n, i, ascending)

    for i in range(n - 1, 0, -1):
        draw_info.lst[i], draw_info.lst[0] = draw_info.lst[0], draw_info.lst[i]
        draw_list(draw_info, {0: draw_info.CURRENT_LIST_POS, i: draw_info.REPLACE_IN_LIST}, True)
        yield True
        yield from heapify(draw_info, i, 0, ascending)

def cocktail_sort(draw_info, ascending=True):
    lst = draw_info.lst
    n = len(lst)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False
        for i in range(start, end):
            freq1 = map_value_to_freq(lst[i])
            freq2 = map_value_to_freq(lst[i + 1])
            play_sound(freq1)
            play_sound(freq2)
            if (lst[i] > lst[i + 1] and ascending) or (lst[i] < lst[i + 1] and not ascending):
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
                swapped = True
                draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, i + 1: draw_info.REPLACE_IN_LIST}, True)
                yield True
        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end, start - 1, -1):
            freq1 = map_value_to_freq(lst[i])
            freq2 = map_value_to_freq(lst[i + 1])
            play_sound(freq1)
            play_sound(freq2)

            if (lst[i] > lst[i + 1] and ascending) or (lst[i] < lst[i + 1] and not ascending):
                lst[i], lst[i + 1] = lst[i + 1], lst[i]
                swapped = True
                draw_list(draw_info, {i: draw_info.CURRENT_LIST_POS, i + 1: draw_info.REPLACE_IN_LIST}, True)
                yield True
        start += 1

def insertion_sort_bucket(bucket, draw_info, ascending=True):
    for i in range(1, len(bucket)):
        current = bucket[i]
        j = i - 1
        while j >= 0 and ((bucket[j] > current and ascending) or (bucket[j] < current and not ascending)):
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = current

    for value in bucket:
        freq = map_value_to_freq(value)
        play_sound(freq)
        position = len(draw_info.lst) - len(bucket) + bucket.index(value)
        draw_list(draw_info, {position: draw_info.REPLACE_IN_LIST}, True)
        yield True

    return bucket

def bucket_sort(draw_info, ascending=True):
    lst = draw_info.lst
    max_value = max(lst)
    bucket_count = 10
    buckets = [[] for _ in range(bucket_count)]

    for value in lst:
        index = int(value / max_value * (bucket_count - 1))
        buckets[index].append(value)

    sorted_list = []

    for bucket in buckets:
        if bucket:
            sorted_bucket = yield from insertion_sort_bucket(bucket, draw_info, ascending)
            sorted_list.extend(sorted_bucket)

            for value in sorted_bucket:
                freq = map_value_to_freq(value)
                play_sound(freq)

            draw_info.lst = sorted_list
            for j in range(len(sorted_list)):
                draw_list(draw_info, {j: draw_info.REPLACE_IN_LIST}, True)
                freq = map_value_to_freq(j)
                play_sound(freq)
                yield True

    draw_info.lst = sorted_list


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

    # Define sorting algorithms and their names
    sorting_algorithms = [
        (bubble_sort, "Bubble Sort"),
        (insertion_sort, "Insertion Sort"),
        (selection_sort, "Selection Sort"),
        (counting_sort, "Counting Sort"),
        (merge_sort, "Merge Sort"),
        (shell_sort, "Shell Sort"),
        (comb_sort, "Comb Sort"),
        (radix_sort, "Radix Sort"),
        (bogo_sort, "Bogo Sort"),
        (quick_sort, "Quick Sort"),
        (heap_sort, "Heap Sort"),
        (cocktail_sort, "Cocktail Sort"),
        (bucket_sort, "Bucket Sort")
    ]

    # Set initial sorting algorithm
    sorting_algorithm, sorting_algo_name = sorting_algorithms[0]
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
                lst = generate_1_to_100_list()
                draw_info.set_list(lst)
                sorting = False
            elif event.key == pygame.K_SPACE and not sorting:
                sorting = True
                if sorting_algorithm == merge_sort or sorting_algorithm == quick_sort:
                    sorting_algorithm_generator = sorting_algorithm(draw_info, 0, len(draw_info.lst) - 1, ascending)
                else:
                    sorting_algorithm_generator = sorting_algorithm(draw_info, ascending)
            elif event.key == pygame.K_a and not sorting:
                ascending = True
            elif event.key == pygame.K_d and not sorting:
                ascending = False
            elif event.key == pygame.K_s and not sorting:
                # Cycle through sorting algorithms
                current_index = sorting_algorithms.index((sorting_algorithm, sorting_algo_name))
                next_index = (current_index + 1) % len(sorting_algorithms)
                sorting_algorithm, sorting_algo_name = sorting_algorithms[next_index]
    pygame.quit()

if __name__ == "__main__":
    main()
