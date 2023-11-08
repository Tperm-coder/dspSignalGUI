import PySimpleGUI as sg
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import math

# Function to quantize the input signal
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2
from signalcompare import SignalComapreAmplitude, SignalComaprePhaseShift


def quantize(signal, levels):
    
    signal_range = np.max(signal) - np.min(signal)
    
    delta = signal_range / levels
    
    midpoints = np.arange(np.min(signal) + delta/2, np.max(signal), delta)    #arange generate array 
    
    quant_signal = np.zeros_like(signal)
    level_number = np.zeros_like(signal, dtype=int)
    encode = []
    for i in range(len(signal)):
        level_number[i] = np.argmin(np.abs(signal[i] - midpoints))
        quant_signal[i] = midpoints[level_number[i]]
        encode.append(level_number[i])
    
    error = quant_signal - signal
   
    t = np.arange(0, len(signal), 1)
    return quant_signal, error, level_number, encode

def apply_fourier_transform(signal, fs, save_file_name):
   
    N = len(signal)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)

    
    amplitude = abs(X)
    phase_shift = np.angle(X)

    # Save at file 
    with open(save_file_name, 'w') as f:
        f.write('0\n')
        f.write('1\n')
        f.write(f'{len(amplitude)}\n')
        for i in range(len(amplitude)):
            f.write(f'{amplitude[i]},{phase_shift[i]}\n')

    
    omega = 2*np.pi/(N*(1/fs))
    freq =np.arange(omega,(omega*(N+1)),omega)

    


    plt.figure()
    plt.stem(freq, amplitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency versus Amplitude Relation')
    plt.show()

    
    plt.figure()
    plt.stem(freq, phase_shift)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Frequency versus Phase Relation')
    plt.show()

    return amplitude, phase_shift

def reconstruct_signal(file_name):
    
    amplitude = []
    phase_shift = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(',')) == 2:
                L = line.split(',')
                V1 = float(L[0].replace('f', ''))
                V2 = float(L[1].replace('f', ''))
                amplitude.append(V1)
                phase_shift.append(V2)
                line = f.readline()
            else:
                break

    # Convert to complex 
    spectrum = [amplitude[i] * np.exp(1j * phase_shift[i]) for i in range(len(amplitude))]

    
    signal = np.zeros(len(spectrum), dtype=np.complex128)
    for n in range(len(spectrum)):
        for k in range(len(spectrum)):
            signal[n] += spectrum[k] * np.exp(2j * np.pi * k * n / len(spectrum))

        signal[n]*=(1/(len(spectrum)))

    return np.real(signal)
    # return np.real(signal)



layout = [
[sg.TabGroup([
    [sg.Tab('Quantization', [[sg.Menu([['Signal Generation', ['Sine Wave', 'Cosine Wave']],
              ['Arithmetic Operations', ['Addition', 'Subtraction', 'Multiplication', 'Squaring', 'Shifting', 'Normalization', 'Accumulation']]])],
    [sg.Text('Click a menu item to generate a signal or perform an arithmetic operation')],
    [sg.Canvas(key='-CANVAS-')],
    [sg.Text('Select a signal file:')],
    [sg.Input(key='file'), sg.FileBrowse()],
    [sg.Text('Enter the number of levels:')],
    [sg.Input(key='levels')],
    [sg.Button('Quantize Signal')],
    [sg.Multiline(size=(60, 10), key='output')]])],

    [sg.Tab('Fourier Transform', [
        [sg.Text('Select a file to save the frequency components in polar form:')],
        [sg.Input(key='save_file'), sg.FileSaveAs()],
        [sg.Text('Select a signal file:')],
        [sg.Input(key='filef'), sg.FileBrowse()],
        [sg.Text('Enter the sampling frequency (in Hz):')],
        [sg.Input(key='fs')],
        [sg.Button('Apply Fourier Transform')],
        [sg.Canvas(key='-CANVAS2-')],
    ])],
    [sg.Tab('Signal Reconstruction', [
        [sg.Text('Select a file to load the frequency components in polar form:')],
        [sg.Input(key='freq_file'), sg.FileBrowse()],
        [sg.Button('Reconstruct Signal')],
        [sg.Canvas(key='-CANVAS4-')],
    ])]

    ])],


]

# Create a window
window = sg.Window('Signal Generator', layout)

# Initialize variables
signal = None
signal_name = None

# Event loop to handle user input
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Sine Wave':
        # Prompt the user for the signal parameters
        A = float(sg.popup_get_text('Enter the amplitude: '))
        theta = float(sg.popup_get_text('Enter the phase shift (in radians): '))
        f = float(sg.popup_get_text('Enter the analog frequency (in Hz): '))
        fs = float(sg.popup_get_text('Enter the sampling frequency (in Hz): '))
        fs = max(fs, 2*f)
        # Generate the time axis
        t = np.arange(0, 1, 1/fs)
        # Calculate the signal
        signal = A * np.sin(2 * np.pi * f * t + theta)
        signal_name = 'Sine Wave'
        # Plot the signal
        plt.plot(t, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.ylim(-3.0, 3.0)
        plt.show()
    elif event == 'Cosine Wave':
        # Prompt the user for the signal parameters
        A = float(sg.popup_get_text('Enter the amplitude: '))
        theta = float(sg.popup_get_text('Enter the phase shift (in radians): '))
        f = float(sg.popup_get_text('Enter the analog frequency (in Hz): '))
        fs = float(sg.popup_get_text('Enter the sampling frequency (in Hz): '))
        # Generate the time axis
        t = np.arange(0, 1, 1/fs)
        # Calculate the signal
        signal = A * np.cos(2 * np.pi * f * t + theta)
        signal_name = 'Cosine Wave'
        # Plot the signal
        plt.plot(t, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.ylim(-3.0, 3.0)
        plt.show()
    elif event == 'Addition':
        # Prompt the user for the number of signals to add
        num_signals = int(sg.popup_get_text('Enter the number of signals to add: '))
        signals = []
        for i in range(num_signals):
            # Prompt the user for the input file
            filename = sg.popup_get_file(f'Select signal {i+1} to add', no_window=True)
            if not filename:
                break
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                signal = []
                while line:
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
                signal = np.array(signal, dtype=float)
                signals.append(signal)
        if len(signals) == 0:
            break
        # Add the signals
        signal = np.sum(signals, axis=0)
        signal_name = 'Addition'
        t = np.arange(0, len(signal), 1)
        plt.plot(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Subtraction':
        signal1 = []
        signal2 = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal1.append(V2)
                        line = f.readline()
                    else:
                        break
                signal1 = np.array(signal1, dtype=float)
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal2.append(V2)
                        line = f.readline()
                    else:
                        break
                signal2 = np.array(signal2, dtype=float)
            # Check if the signals have the same length
            if len(signal1) != len(signal2):
                sg.popup_error('The signals must have the same length')
                continue
        # Subtract the signals
        signal = signal1 - signal2
        signal_name = 'Subtraction'
        # Plot the signal
        t = np.arange(0, len(signal), 1)
        plt.plot(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Multiplication':
        signal = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
                signal = np.array(signal, dtype=float)
        constant = float(sg.popup_get_text('Enter the constant: '))
        # Multiply the signal by the constant
        signal = signal * constant
        signal_name = 'Multiplication'
        # Plot the signal
        t = np.arange(0, len(signal), 1)
        plt.plot(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Squaring':
        signal = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
                signal = np.array(signal, dtype=float)
        # Square the signal
        signal = signal ** 2
        signal_name = 'Squaring'
        # Plot the signal
        t = np.arange(0, len(signal), 1)
        plt.plot(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Shifting':
        signal = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[0])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
                signal = np.array(signal, dtype=float)
        constant = float(sg.popup_get_text('Enter the constant: '))
        # Shift the signal
        signal = signal + constant
        signal_name = 'Shifting'
        # Plot the signal
        t = np.arange(0, len(signal), 1)
        plt.plot(signal, t)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Normalization':
        signal = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
                signal = np.array(signal, dtype=float)
        norm_type = sg.popup_get_text('Enter the normalization type 1 for (0 to 1) or -1 (-1 to 1): ')
        # Normalize the signal
        if norm_type == '1':
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        elif norm_type == '-1':
            signal = (2 * signal - np.max(signal) - np.min(signal)) / (np.max(signal) - np.min(signal))
        signal_name = 'Normalization'
        # Plot the signal
        t = np.arange(0, len(signal), 1)
        plt.plot(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title(signal_name)
        plt.show()
    elif event == 'Accumulation':
        signal = []
        # Prompt the user for the input file
        filename = sg.popup_get_file('Select a file to open', no_window=True)
        if filename:
            # Read the file and extract the signal data
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                        line = f.readline()
                    else:
                        break
            signal = np.array(signal, dtype=float)
            # Accumulate the signal
            signal = np.cumsum(signal)
            signal_name = 'Accumulation'
            # Plot the signal
            t = np.arange(0, len(signal), 1)
            plt.plot(t, signal)
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.title(signal_name)
            plt.show()
    elif event == 'Quantize Signal':
        # Read the file and extract the signal data
        filename = values['file']
        levels = int(values['levels'])
        if filename and levels > 0:
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                signal = []
                while line:
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                    else:
                        break
                    line = f.readline()
            signal = np.array(signal, dtype=float)
            # Quantize the signal
            quant_signal, error, encode, level = quantize(signal, levels)
            bits = math.log2(levels)
            string = '0' + str(int(bits)) + 'b'
            encode = [format(n, string) for n in level]
            level = [n + 1 for n in level]
            # Plot the quantized signal
            t = np.arange(0, len(quant_signal), 1)
            plt.stem(t, quant_signal)
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.title('Quantized Signal')
            plt.show()
            # Display the quantization error in the output textbox
            output = f'Quantization Error:\n{error}\n\nEncoded Signal:\n{encode}'
            window['output'].update(output)
            print(level)
            print(encode)
            print(error)
            #test1
            QuantizationTest1('Quan1_Out.txt', encode, quant_signal)
            #test2
            QuantizationTest2('Quan2_Out.txt',level, encode, quant_signal,error)
        else:
            sg.popup('Please enter a valid signal file and number of levels.')

    elif event == 'Apply Fourier Transform':
        # Read the file and extract the signal data
        filename = values['filef']
        save_file_name = values['save_file']
        signal = []
        if filename:
            with open(filename, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    L = line.strip()
                    if len(L.split(' ')) == 2:
                        L = line.split(' ')
                        V2 = float(L[1])
                        signal.append(V2)
                    else:
                        break
                    line = f.readline()
            signal = np.array(signal, dtype=float)
        if signal is not None:
            fs = float(values['fs'])
            sig, phase = apply_fourier_transform(signal, fs, save_file_name)

            filename = 'Output_Signal_DFT_A,Phase.txt'
            testamp = []
            testphase = []
            if filename:
                with open(filename, 'r') as f:
                    line = f.readline()
                    line = f.readline()
                    line = f.readline()
                    line = f.readline()
                    while line:
                        L = line.strip()
                        if len(L.split(' ')) == 2:
                            L = line.split(' ')
                            V1 = float(L[0].replace('f', ''))
                            V2 = float(L[1].replace('f', ''))
                            testamp.append(V1)
                            testphase.append(V2)
                        else:
                            break
                        line = f.readline()
                testamp = np.array(testamp, dtype=float)
                testphase = np.array(testphase, dtype=float)


                amp = SignalComapreAmplitude(sig, testamp)
                phs = SignalComaprePhaseShift(phase, testphase)

                # print((sig[1]))
                # print((20.90500744 - 20.90500744) > 0.001)
                # print((20.90500744 - 20.90500744))
                # print(sig[1] + testamp[1])
                print(testamp)

                if not amp:
                    print("error in amplitude")
                elif not phs:
                    print("error in phase")
                else:
                    print("test case passes successfully")

        else:
            sg.popup('Please generate or load a signal first.')

    elif event == 'Reconstruct Signal':
        # Read 
        file_name = values['freq_file']

        
        signal = reconstruct_signal(file_name)

        # Plot the reconstructed signal
        t = np.arange(0, len(signal), 1)
        plt.stem(t, signal)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('Reconstructed Signal')
        plt.show()