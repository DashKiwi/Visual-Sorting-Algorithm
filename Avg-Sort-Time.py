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

def bubble_sort(arr):
    for i in range(len(arr) - 1, 0, -1):
        for j in range(i):
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def insertion_sort(arr):
    n = len(arr)
    if n <= 1:
        return
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

def selection_sort(arr):
    size = len(arr)
    for i in range(size):
        min_index = i
        for j in range(i + 1, size):
            if arr[j] < arr[min_index]:
                min_index = j
        (arr[i], arr[min_index]) = (arr[min_index], arr[i])

def count_sort(arr):
    max_value = max(arr) if arr else 0
    count = [0] * (max_value + 1)

    for num in arr:
        count[num] += 1

    output = []
    for num, cnt in enumerate(count):
        output.extend([num] * cnt)

    for i in range(len(arr)):
        arr[i] = output[i]


def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j-gap] > temp:
                arr[j] = arr[j-gap]
                j -= gap
            arr[j] = temp
        gap //= 2

def counting_sort_radix(arr, exp1): 
    n = len(arr) 
    output = [0] * (n) 
    count = [0] * (10) 
    
    for i in range(0, n): 
        index = (arr[i]/exp1) 
        count[int((index)%10)] += 1

    for i in range(1,10): 
        count[i] += count[i-1] 

    i = n-1
    while i>=0: 
        index = (arr[i]/exp1) 
        output[ count[ int((index)%10) ] - 1] = arr[i] 
        count[int((index)%10)] -= 1
        i -= 1

    i = 0
    for i in range(0,len(arr)): 
        arr[i] = output[i] 

def radix_sort(arr):
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        counting_sort_radix(arr,exp)
        exp *= 10

def comb_sort(arr):
    shrink_factor = 1.3
    gap = len(arr)
    completed = False

    while not completed:
        gap = int(gap / shrink_factor)
        if gap <= 1:
            completed = True

        index = 0
        while index + gap < len(arr):
            if arr[index] > arr[index + gap]:
                arr[index], arr[index + gap] = arr[index + gap], arr[index]
                completed = False
            index += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def cocktail_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1

    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        if not swapped:
            break

        swapped = False
        end -= 1

        for i in range(end-1, start-1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True

        start += 1

def bucket_sort(arr):
    bucket = []
    num_buckets = 10

    for _ in range(num_buckets):
        bucket.append([])

    for num in arr:
        index = int(num * num_buckets) // (max(arr) + 1)
        bucket[index].append(num)

    sorted_arr = []
    for b in bucket:
        sorted_arr.extend(sorted(b))

    return sorted_arr

def average_sort_time(algorithm, n_runs, array_size, value_range):
    total_time = 0
    print(f"\nStarting {algorithm}")
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

def run_all_algorithms(n_runs, array_size, value_range):
    algorithms = {
        "Merge Sort": merge_sort,
        "Bubble Sort": bubble_sort,
        "Insertion Sort": insertion_sort,
        "Selection Sort": selection_sort,
        "Quick Sort": quick_sort,
        "Count Sort": count_sort,
        "Shell Sort": shell_sort,
        "Radix Sort": radix_sort,
        "Comb Sort": comb_sort,
        "Heap Sort": heap_sort,
        "Cocktail Sort": cocktail_sort,
        "Bucket Sort": bucket_sort,
    }

    print("\nRunning all algorithms...")
    for name, algorithm in algorithms.items():
        avg_time = (average_sort_time(algorithm, n_runs, array_size, value_range) * 1000)
        print(f"Average time taken by {name}: {avg_time:.6f} milliseconds")

def settings_choices():
    settings_choice = int_checker(
                "Would you like to use\n"
                "1. Default Settings (1,000 runs, array size of 10,000, value range of 1 - 10,000)\n"
                "2. Custom settings\n", 2)

    if settings_choice == 1:
        n_runs = 1000
        array_size = 10000
        value_range = 10000
    else:
        n_runs = int_checker("Please enter the amount of tests you want to run: ", float('inf'))
        array_size = int_checker("Please enter the array size for each test: ", float('inf'))
        value_range = int_checker("Please enter a value range for the array: ", float('inf'))

def main():
    while True:
        algorithm_choice = int_checker(
            "Please choose which sorting algorithm you would like to test:\n"
            "1. Merge Sort\n"
            "2. Bubble Sort\n"
            "3. Insertion Sort\n"
            "4. Selection Sort\n"
            "5. Quick Sort\n"
            "6. Count Sort\n"
            "7. Shell Sort\n"
            "8. Radix Sort\n"
            "9. Comb Sort\n"
            "10. Heap Sort\n"
            "11. Cocktail Sort\n"
            "12. Bucket Sort\n"
            "13. Run All Algorithms\n"
            "14. Exit\n", 14)

        if algorithm_choice in range(1, 13):
            algorithm_map = {
                1: merge_sort,
                2: bubble_sort,
                3: insertion_sort,
                4: selection_sort,
                5: quick_sort,
                6: count_sort,
                7: shell_sort,
                8: radix_sort,
                9: comb_sort,
                10: heap_sort,
                11: cocktail_sort,
                12: bucket_sort,
            }
            algorithm = algorithm_map[algorithm_choice]
            algorithm_name = list(algorithm_map.keys())[algorithm_choice - 1]
            
            settings_choice = int_checker(
                "Would you like to use\n"
                "1. Default Settings (1,000 runs, array size of 10,000, value range of 1 - 10,000)\n"
                "2. Custom settings\n", 2)

            if settings_choice == 1:
                n_runs = 1000
                array_size = 10000
                value_range = 10000
            else:
                n_runs = int_checker("Please enter the amount of tests you want to run: ", float('inf'))
                array_size = int_checker("Please enter the array size for each test: ", float('inf'))
                value_range = int_checker("Please enter a value range for the array: ", float('inf'))

            average_time = (average_sort_time(algorithm, n_runs, array_size, value_range) * 1000)
            print(f"Average time taken to sort a random array with {algorithm_name}: {average_time:.6f} milliseconds\n"
                  f"{n_runs} arrays sorted each array containing {array_size} items between 1 and {value_range}")

        elif algorithm_choice == 13:
            settings_choice = int_checker(
                "Would you like to use\n"
                "1. Default Settings (1,000 runs, array size of 10,000, value range of 1 - 10,000)\n"
                "2. Custom settings\n", 2)

            if settings_choice == 1:
                n_runs = 1000
                array_size = 10000
                value_range = 10000
            else:
                n_runs = int_checker("Please enter the amount of tests you want to run: ", float('inf'))
                array_size = int_checker("Please enter the array size for each test: ", float('inf'))
                value_range = int_checker("Please enter a value range for the array: ", float('inf'))

            run_all_algorithms(n_runs, array_size, value_range)

        elif algorithm_choice == 14:
            break

main()