import tkinter as tk
from tkinter import filedialog
import numpy as np

from task_6.DerivativeSignal import DerivativeSignal

from task_6.TestCases.shift_or_fold.Shift_Fold_Signal import Shift_Fold_Signal

from task_6.comparesignal2 import SignalSamplesAreEqual
import math


def show_task_6_window():
    top_window = tk.Toplevel()
    top_window.geometry('500x500')
    top_window.title("Task 6")
    Smoothing_btn = tk.Button(top_window, text="Smoothing", command=lambda: show_smothing_window())
    Sharpening_btn = tk.Button(top_window, text="Sharpening", command=lambda: Sharpening())
    DelayingOrAdvancing_btn = tk.Button(top_window, text="Delaying Or Advancing",
                                        command=lambda: show_delay_advance_signal())
    Folding_btn = tk.Button(top_window, text="Folding", command=lambda: Folding())

    RemovetheDC_Component_btn = tk.Button(top_window, text="Remove the DC component",
                                          command=lambda: RemovetheDC_Component(
                                              './task_6/TestCases/Remove Dc/DC_component_input.txt'))
    Smoothing_btn.pack(padx=10, pady=10)
    Sharpening_btn.pack(padx=10, pady=10)
    DelayingOrAdvancing_btn.pack(padx=10, pady=10)
    Folding_btn.pack(padx=10, pady=10)
    RemovetheDC_Component_btn.pack(padx=10, pady=10)
    top_window.mainloop()


def parse_input(signal_file_path):
    signal = []
    time = []
    with open(signal_file_path, 'r') as file:
        signal_data = file.read().split('\n')
        for i in range(3, len(signal_data)):
            line = signal_data[i].split(' ')
            time.append(int(line[0]))
            signal.append(float(line[1]))
    return time, signal


def choose_file():
    file_path = filedialog.askopenfilename(title="Select a file",
                                           filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    return file_path

def show_smothing_window():
    top_window = tk.Toplevel()
    top_window.geometry('500x500')
    top_window.title("Smoothing Input")

    input_label = tk.Label(top_window, text="Enter Window Size : ")
    input_label.pack()

    window_size_entry = tk.Entry(top_window)
    window_size_entry.pack()

    # Create a variable to capture the file path
    chosen_file_path = tk.StringVar()

    def set_chosen_file_path():
        chosen_file_path.set(choose_file())

    file_path_btn = tk.Button(top_window, text="Choose File", command=lambda: set_chosen_file_path())
    file_path_btn.pack()

    smooth_signal_btn = tk.Button(top_window, text="Smooth Signal",
                                  command=lambda: smooth_signal(int(window_size_entry.get()), chosen_file_path.get()))
    smooth_signal_btn.pack()

    top_window.mainloop()

def show_delay_advance_signal():
    top_window = tk.Toplevel()
    top_window.geometry('500x500')
    top_window.title("Delay Advance Signal")

    k_value_label = tk.Label(top_window, text="Enter K : ")
    k_value_label.pack()

    k_value_entry = tk.Entry(top_window)
    k_value_entry.pack()

    # Create a variable to capture the file path
    chosen_file_path = tk.StringVar()

    def set_chosen_file_path():
        chosen_file_path.set(choose_file())

    file_path_btn = tk.Button(top_window, text="Choose File", command=lambda: set_chosen_file_path())
    file_path_btn.pack()

    advance_delay_signal_btn = tk.Button(top_window, text="Advance_Delay Signal",
                                         command=lambda: DelayingOrAdvancing(int(k_value_entry.get()),
                                                                             chosen_file_path.get()))
    advance_delay_signal_btn.pack()

    top_window.mainloop()

def smooth_signal(window_size, signal_file_path):
    time, signal = parse_input(signal_file_path)
    ans = []
    N = len(signal)
    for i in range(N):
        sum = 0
        for j in range(i, min(i + window_size, N)):
            sum += signal[j]
        ans.append(sum / window_size)
    return ans

def Sharpening():
    DerivativeSignal()

def DelayingOrAdvancing(val, signal_file_path):
    time, signal = parse_input(signal_file_path)
    for i in range(len(time)):
        time[i] += val
    if val == -500:
        Shift_Fold_Signal("./task_6/TestCases/shift_or_fold/Output_ShiftFoldedby-500.txt", time, signal)
    elif val == 500:
        Shift_Fold_Signal("./task_6/TestCases/shift_or_fold/Output_ShifFoldedby500.txt", time, signal)

def Folding():
    signal_file_path = './task_6/TestCases/shift_or_fold/input_fold.txt'
    time, signal = parse_input(signal_file_path)
    ans = signal
    ans.reverse()
    Shift_Fold_Signal("./task_6/TestCases/shift_or_fold/Output_fold.txt", time, ans)

def RemovetheDC_Component(signal_file_path):
    time, signal = parse_input(signal_file_path)
    amplitude, phase_shift = calcDFT(time, signal)
    new_time, new_signal = calcIDFT(amplitude, phase_shift)
    N = len(new_signal)
    for i in range(N):
        print(new_signal[i])
    SignalSamplesAreEqual('./task_6/TestCases/Remove Dc/DC_component_output.txt', new_signal)

def calcDFT(time, signal):
    N = len(signal)
    amplitude = []
    phase_shift = []
    for k in range(N):
        real_number = 0
        imaginary_number = 0
        for n in range(N):
            theta = (2 * np.pi * k * n) / N
            real_part = np.cos(theta)
            imaginary_part = -np.sin(theta)
            real_number += signal[n] * real_part
            imaginary_number += signal[n] * imaginary_part
        dft_result = complex(real_number, imaginary_number)
        amplitude.append(abs(dft_result))
        phase_shift.append(np.angle(dft_result))
    amplitude[0] = 0
    phase_shift[0] = 0
    return amplitude, phase_shift


def calcIDFT(amplitude, phase_shift):
    N = len(phase_shift)
    complex_signal = []
    for i in range(N):
        real_part = amplitude[i] * np.cos(phase_shift[i])
        imaginary_part = amplitude[i] * np.sin(phase_shift[i])
        complex_signal.append(complex(real_part, imaginary_part))
    N = len(complex_signal)
    original_signal = []
    time = []
    for n in range(N):
        accumulator = 0
        time.append(n)
        for k in range(N):
            theta = (2 * np.pi * k * n) / N
            real_part = np.cos(theta)
            imaginary_part = np.sin(theta)
            current_complex = complex(real_part, imaginary_part)
            result = complex_signal[k] * current_complex
            real_temp = round(result.real, 2)
            imag_temp = round(result.imag, 2)
            accumulator += complex(real_temp, imag_temp)
        original_signal.append((accumulator.real / N))
    return time, original_signal
