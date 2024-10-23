import random
import time

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def average_sort_time(algorithm, n_runs, array_size, value_range):
    total_time = 0
    print("Starting")
    for _ in range(n_runs):
        random_array = [random.randint(1, value_range) for _ in range(array_size)]
        start_time = time.time()
        algorithm(random_array)
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / n_runs
    return average_time

def int_checker(message, accepted):
    while True:
        try:
            num = int(input(message))
            if num < 1 or (not isinstance(accepted, (int, float)) and accepted is not float('inf') and num > int(accepted)):
                raise ValueError(f"Number must be between 1 and {accepted}.")
            return num
        except ValueError as e:
            print(f"Please enter a valid integer: {e}\n")

def main():
    while True:
        algorithm_choice = int_checker("Please choose which sorting algorithm you would like to test:\n1. Merge Sort\n2. Exit\n", 2)
        if algorithm_choice == 1:
            algorithm = merge_sort
            algorithm_name = "Merge Sort"
        elif algorithm_choice == 2:
            break
        settings_choice = int_checker("Would you like to use\n1. Default Settings (1.000 runs, array size of 10.000, value range of 1 - 10.000)\n2. Custom settings\n", 2)
        if settings_choice == 1:
            n_runs = 1000
            array_size = 10000
            value_range = 10000
            average_time = average_sort_time(algorithm, n_runs, array_size, value_range)
        elif settings_choice == 2:
            n_runs = int_checker("Please enter the amount of tests you want to run: ", float('inf'))
            array_size = int_checker("Please enter the array size for each test: ", float('inf'))
            value_range = int_checker("Please enter a value range for the array: ", float('inf'))
            average_time = average_sort_time(algorithm, n_runs, array_size, value_range)

        print(f"Average time taken to sort a random array with {algorithm_name}: {average_time:.6f} seconds\n{n_runs} arrays sorted each array containing {array_size} items between 1 and {value_range}")

main()