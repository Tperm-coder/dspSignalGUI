import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import tkinter as tk
import math
import comparesignal2


# def read_input(signal_path):
#     with open(signal_path, "r") as file:
#         lines = file.readlines()[3:]  # Read all lines excluding the first 3 rows
#
#     # Extract x and y values from the lines
#     x_values = []
#     y_values = []
#     for line in lines:
#         values = line.strip().split()
#         x_values.append(float(values[0]))
#         y_values.append(float(values[1]))
#     file.close()
#     return x_values, y_values
#
#
# def load_signal():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         x_values, y_values = read_input(file_path)
#         return x_values, y_values
#     else:
#         print("No file selected.")
#
#
def continuous_representation(x_values, y_values):
    plt.plot(x_values, y_values)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Signal")
    plt.show()


def calculate_dct(x_values, y_values, m):
    # mean = np.mean(y_values)
    # y_values_centered = y_values - mean

    N = len(y_values)
    dct_values = np.zeros(N)

    for k in range(N):
        for n in range(N):
            coefficient = math.sqrt(2 / N) * y_values[n] * math.cos((math.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
            dct_values[k] += coefficient

    dct_values = np.round(dct_values, decimals=6)
    # Keep only the first m coefficients and set the rest to zero
    dct_values[m:] = 0
    filepath = "DCT_output.txt"
    comparesignal2.SignalSamplesAreEqual(filepath, dct_values)
    return dct_values


def compute_dct(m):
    x_values, y_values = load_signal()

    dct_coefficients = calculate_dct(x_values, y_values, m)
    output = []

    print("DCT Coefficients:")
    for i, coefficient in enumerate(dct_coefficients):
        output.append(f"{i} {coefficient}")
        print(f"{i} {coefficient}")

    # Display the DCT result
    plt.plot(dct_coefficients)
    plt.xlabel("Index")
    plt.ylabel("DCT Coefficient")
    plt.title("Discrete Cosine Transform")
    plt.show()

    return output


def save_dct_output(dct_coefficients):
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        with open(file_path, "w") as file:
            for coef in dct_coefficients:
                file.write(str(coef) + "\n")
        print("DCT coefficients saved successfully.")
        messagebox.showinfo("Save", "DCT coefficients saved successfully.")
    else:
        print("Save cancelled.")
        messagebox.showinfo("Save", "Save cancelled.")

#
# def remove_DC():
#     x_values, y_values = load_signal()
#     sum = 0
#     for i in range(len(y_values)):
#         sum = sum + y_values[i]
#     average = sum / len(y_values)
#     new_y = []
#     for n in range(len(y_values)):
#         y = y_values[n] - average
#         new_y.append(y)
#     filepath = 'DC_component_output.txt'
#     print("Signal after removing DC component:")
#     print(new_y)
#     plt.plot(x_values, y_values)
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Signal")
#     plt.show()


##############################################
window = tk.Tk()
window.title('Sin and Cos Signal Generator')

# Adjusted geometry to better fit the content
window.geometry("400x250")


def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        x_values, y_values = read_input(file_path)
        continuous_representation(x_values, y_values)


def compute_dct_from_input():
    m = int(entry.get())
    dct_coefficients = compute_dct(m)
    save_dct_output(dct_coefficients)


# Create the buttons and textbox
open_button = tk.Button(window, text="Open File", command=open_file_dialog)
open_button.pack(pady=10)

label = tk.Label(window, text="Enter the number of coefficients to keep:")
label.pack()

entry = tk.Entry(window)
entry.pack()

compute_dct_button = tk.Button(window, text="Compute DCT", command=compute_dct_from_input)
compute_dct_button.pack(pady=10)

remove_DC_button = tk.Button(window, text="Remove DC Component", command=remove_DC)
remove_DC_button.pack(pady=10)

window.mainloop()
