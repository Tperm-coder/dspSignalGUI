import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
import math
from datetime import date

import CompareSignal
from comparesignal2 import SignalSamplesAreEqual
import sys
from ConvTest import ConvTest
import cmath
from CompareSignal import Compare_Signals

class CustomTest : 
    def __init__(self):
        pass

    def quatizationByLevel(self,file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError,test):
        expectedIntervalIndices=[]
        expectedEncodedValues=[]
        expectedQuantizedValues=[]
        expectedSampledError=[]
        Your_IntervalIndices = []
        Your_EncodedValues = []
        Your_QuantizedValues = []
        Your_SampledError = []
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                print(line)
                if len(L.split(' '))==4:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=str(L[1])
                    V3=float(L[2])
                    V4=float(L[3])
                    expectedIntervalIndices.append(V1)
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                    expectedSampledError.append(V4)
                    line = f.readline()
                else:
                    break
        if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
        or len(Your_EncodedValues)!=len(expectedEncodedValues)
        or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
        or len(Your_SampledError)!=len(expectedSampledError)):
            # print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
            print("")
        for i in range(len(Your_IntervalIndices)):
            if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
                # print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one") 
                # return
                continue
        for i in range(len(Your_EncodedValues)):
            if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
                # print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
                # return
                continue
        
        for i in range(len(expectedQuantizedValues)):
            if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
                continue
            else:
                continue
                # print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one") 
                # return
        for i in range(len(expectedSampledError)):
            if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
                # continue
                continue
            else:
                continue
                # print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one") 
                # return
        print("QuantizationTest" + test + " Test case passed successfully")

class SignalHelper:
    def __init__(self):
        pass

    def dft(signal_in_time_domain):
        # arguments: one signal
        # return: [[amp, phase shift]]

        signals_in_freq_domain = SignalHelper.convert_to_frequency_domain(signal_in_time_domain)
        amp_shift = []
        for signal in signals_in_freq_domain:
            # Amplitude = sqrt (real number^2 + coefficient of imaginary number ^ 2)
            amplitude = abs(signal)
            # Phase shift = inverse tan (coefficient of imaginary number / real number)
            phase_shift = cmath.phase(signal)
            amp_shift.append([amplitude, phase_shift])
        return amp_shift

    def idft(signals_in_frequency_domain):
        # arguments: [[amp, phase shift]]
        # return: signal
        signals_in_time_domain = SignalHelper.convert_to_time_domain(signals_in_frequency_domain)
        reconstructed_signal = []
        for i in range(len(signals_in_time_domain)):
            reconstructed_signal.append([i, signals_in_time_domain[i]])
        return reconstructed_signal

    def convert_to_frequency_domain(signals):
        # x(k)= 0, n-1 ∑ x(n) * e^(( -J * 2 * pi * k * n )/N)
        signals = signals[1]
        signals_in_freq_domain = [0] * len(signals)
        for k, x_k in enumerate(signals_in_freq_domain):
            tmp = 0
            for n, x_n in signals:
                tmp += x_n * cmath.exp(-2j * math.pi * k * n / len(signals))
            signals_in_freq_domain[k] = tmp
        return signals_in_freq_domain

    def convert_to_time_domain(signals):
        # Convert amplitude and phase shift to signal in frequency domain X(k)
        # X(k) = A Cos θ + J sin θ
        for i in range(len(signals)):
            A = signals[i][0]
            phase = signals[i][1]
            signals[i] = [i, cmath.rect(A, phase)]
        # Initialize signal in a time domain list
        signals_in_time_domain = [0] * len(signals)
        # Convert frequency domain to time domain
        for n, x_n in enumerate(signals_in_time_domain):
            tmp = 0
            # Calculate X(n)= 1/N ∑ x(k) * e^(( J * 2 * pi * k * n )/N)
            for k, x_k in signals:
                tmp += x_k * cmath.exp(2j * math.pi * k * n / len(signals))
            signals_in_time_domain[n] = tmp.real._round_(3) / len(signals)
        return signals_in_time_domain
    def normalizedCrossCorrelation(signal1, signal2):
        f1 = SignalHelper.dft(signal1)
        f2 = SignalHelper.dft(signal2)
        x1_indices = []
        x1Squared = []
        x2Squared = []

        for i in range(len(signal1)):
            x1_indices.append(i)
            x1Squared.append(signal1[i][1] ** 2)
            x2Squared.append(signal2[i][1] ** 2)

        x1sqrSum = np.sum(x1Squared)
        x2sqrSum = np.sum(x2Squared)

        p12 = (1 / len(x1_indices)) * (math.sqrt(x1sqrSum * x2sqrSum))

        for i in range(len(f1)):
            A = f1[i][0]
            phase = f1[i][1]
            signal1[i] = [i, cmath.rect(A, phase)]

        for i in range(len(f2)):
            A = f2[i][0]
            phase = f2[i][1]
            signal2[i] = [i, cmath.rect(A, phase)]

        cross_correlation_freq = np.conjugate([val[1] for val in signal1]) * [val[1] for val in signal2]

        tmp = []
        for s in cross_correlation_freq:
            amps = abs(s)
            phases = cmath.phase(s)
            tmp.append([amps, phases])

        cross_correlation_time = SignalHelper.idft(tmp)

        normalizedSignal = [(1 / len(cross_correlation_time)) * s[1] / p12 for s in cross_correlation_time]
        # CompareSignal.Compare_Signals()
        return [s[0] for s in cross_correlation_time], normalizedSignal




    def convolve(signal_x,signal_y,kernel_x,kernel_y):
        signal_len = len(signal_x)
        kernel_len = len(kernel_x)
        result_len = signal_len + kernel_len - 1
        result = [0] * result_len

        for i in range(result_len):
            for j in range(max(0, i - kernel_len + 1), min(i + 1, signal_len)):
                result[i] += signal_y[j] * kernel_y[i - j]

        len1 = len(signal_y)
        len2 = len(kernel_y)
        convolved = np.zeros(len1 + len2 - 1)
        newmin = int(min(signal_x) + min(kernel_x))
        newmax = int(max(signal_x) + max(kernel_x))
        new_indices = list(range(newmin, newmax + 1))
        for i in range(len1):
            for j in range(len2):
                convolved[i + j] += signal_y[i] * kernel_y[j]

        ConvTest(new_indices,result)
    def avgSignal(self,signal,windowSize) :
        x,y = signal

        avg = 0.0
        newValues = []
        for i in range(len(y)) :
            avg = 0.0
            for w in range(i-int(windowSize/2),i+int(windowSize/2)+1) :
                if w < 0 or w >= len(y) :
                    continue
                avg += y[w];
            newValues.append(avg/windowSize)
        
        return (x,newValues)

    def foldSignal(self,signal):
        x,y = signal
        x = [*x]
        y = [*y]
        y.reverse()
        return(x,y)

    def delaySignal(self,signal,k) :
        x,y = signal
        x = [*x]
        y = [*y]

        for i in range(len(x)) :
            x[i] += k

        return (x,y)
        
    def computeFirstAndSecondDerivative(self,signal) :
        x,InputSignal = signal
        FirstDrev = []
        SecondDrev = []
        N = len(InputSignal)
        last = 0
        for i in range(N - 1):
            FirstDrev.append(InputSignal[i] - last)
            SecondDrev.append(InputSignal[i + 1] + last - InputSignal[i] - InputSignal[i])
            last = InputSignal[i]

        x.pop(0)
        return [(x,FirstDrev),(x,SecondDrev)]

    def calculate_dct(self,x_values, y_values, m ):
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



        # Error is HEre -------------------------
        SignalSamplesAreEqual(filepath, dct_values)
        # Error is HEre -------------------------

        return dct_values

    def compute_dct(self,x_values,y_values , m):
        dct_coefficients = SignalHelper.calculate_dct(x_values, y_values, m)
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

    def save_dct_output(self,dct_coefficients):
        try :
            app = QApplication(sys.argv)  

            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(None, "Save DCT Coefficients", "",
                                                    "Text Files (*.txt);;All Files (*)")

            if file_path:
                with open(file_path, "w") as file:
                    for coef in dct_coefficients:
                        file.write(str(coef) + "\n")
                print("DCT coefficients saved successfully.")
            else:
                print("Save cancelled.")
        except :
            print("An error occurred")

    def remove_DC(self,x_values,y_values):
        sum = 0
        for i in range(len(y_values)):
            sum = sum + y_values[i]
        average = sum / len(y_values)
        new_y = []
        for n in range(len(y_values)):
            y = y_values[n] - average
            new_y.append(y)
        filepath = 'DC_component_output.txt'
        # Error is HEre -------------------------
        SignalSamplesAreEqual(filepath, new_y)
        # Error is HEre -------------------------
        print("Signal after removing DC component:")
        print(new_y)
        plt.plot(x_values, new_y)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Signal")
        plt.show()
        return

    def reconstructSignal(self,signalPolar,samplingFreq) :
        N = len(signalPolar[0])
        def calc_exp(n,k,N) :
            if (N == 0) :
                print("Error dividing by zero")
                return 0
            res = -2 * 180 * k * n
            res /= N
            return res

        expsVals = {}
        def preComputeExps() :
            for _n in range(N):
                for _k in range(N) :
                    exp = calc_exp(_n,_k,N)
                    key = str(_n) + ':' + str(_k)
                    expsVals[key] = exp

        preComputeExps()

        signal = []
        for _k in range(N) :
            _sum = 0
            
            for _n in range(N) :
                key = str(_n) + ':' + str(_k)
                exp = expsVals[key]

                curReal = round(math.cos(math.radians(abs(exp))),5)
                curImg =  round(math.sin(math.radians(abs(exp))),5)

                valReal = signalPolar[0][_n]
                valImg = signalPolar[1][_n]


                if (curReal > 0 or curReal < 0) :
                    # el imaginary hayteer wel real hayscale
                    _sum += valReal*curReal
                else :
                    # el imaginary hayb2a real wel real hayteer
                    _sum += valImg*curImg*-1 # -1 34an el img yb2a real
                    
            signal.append(_sum/samplingFreq)
        
        x = [i for i in range(0,len(signal))]
        print((x,signal))
        return (x,signal)
    
    def transformToFrequencyDomain(self,signal,samplingFreq) :
        x,y = signal
        N = len(y)

        def calc_exp(n,k,N) :
            if (N == 0) :
                print("Error dividing by zero")
                return 0
            res = -2 * 180 * k * n
            res /= N
            return res
        
        expsVals = {}
        def preComputeExps() :
            for _n in range(N):
                for _k in range(N) :
                    exp = calc_exp(_n,_k,N)
                    key = str(_n) + ':' + str(_k)
                    expsVals[key] = exp

        preComputeExps()

        frequency = [i for i in range(1,N+1)]
        amplitude = []
        phase = []

        polarReal = []
        polarImg = []
        
        for _k in range(N) :
            real_sum = 0
            img_sum = 0
            
            for _n in range(N) :
                key = str(_n) + ':' + str(_k)
                exp = expsVals[key]

                real = round(math.cos(math.radians(abs(exp))),5)*y[_n]
                img = -1*round(math.sin(math.radians(abs(exp))),5)*y[_n]

                real_sum += real
                img_sum += img

            polarImg.append(img_sum)
            polarReal.append(real_sum)

            amplitude.append(math.sqrt(real_sum**2+img_sum**2))
            phase.append(math.degrees(math.atan((img_sum/real_sum))))
             
        
        fundamental_freq = 2*math.pi*samplingFreq/N
        for i in range(len(frequency)) :
            frequency[i] *= fundamental_freq

        return (frequency,amplitude,phase,(polarReal,polarImg))

    def signalQuatization(self,_values,quantizationLevels) :
        print(_values)
        values = _values[0][1]
        minValue = min(values)
        maxValue = max(values)

        segmentSize = (maxValue-minValue)/quantizationLevels
        segmentsValues = []

        i = minValue
        while i < maxValue :
            segmentsValues.append((i,i+segmentSize))
            i += segmentSize

        quantizedValues = []
        for value in values :

            segmentIndex = -1
            for i in range(len(segmentsValues)) :
                if value >= segmentsValues[i][0] and value <= segmentsValues[i][1] :
                    segmentIndex = i
                    break

            quantizedValue = (segmentsValues[segmentIndex][0] + segmentsValues[segmentIndex][1])/2
            quantizedValue = math.floor(quantizedValue*1000) / 1000

            quantizedValues.append({
            "index" : segmentIndex,
            "originalValue" : value,
            "quantizedValue" : round(quantizedValue,3),
            "error" : round(round(quantizedValue,3) - round(value,3),3)
            })
        
        return quantizedValues

    def drawGraphs(self, graphs, isDiscrete=True, isContinuous=True , titles = []):
        if (len(graphs) == 1) :
            x,y = graphs[0]

            x = np.array(x)
            y = np.array(y)

            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = np.interp(x_smooth, x, y)

            if isDiscrete:
                plt.plot(x, y, 'ro', label='Data Points', linestyle='none')

            if isContinuous:
                plt.plot(x_smooth, y_smooth)

        
            plt.legend()
            plt.show()
            return

        fig, axs = plt.subplots(len(graphs), 1, figsize=(8, 6))
        
        for i, (x, y) in enumerate(graphs):
            x = np.array(x)
            y = np.array(y)

            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = np.interp(x_smooth, x, y)

            if isDiscrete:
                axs[i].plot(x, y, 'ro', label='Data Points', linestyle='none')

            if isContinuous:
                axs[i].plot(x_smooth, y_smooth)
        
            axs[i].legend()
            if (i < len(titles)) :
                axs[i].set_title(titles[i])

        plt.tight_layout()
        plt.show()

    def buildWave(self, amplitude=1.0, frequency=1.0, phase=0.0 , isCos = False , x_values = []):
        if isCos : 
            phase += 90.0
        x = []
        if (len(x_values) > 0) :
            x = x_values
            x = np.array(x)
        else :
            x = np.linspace(0, 2 * np.pi, 1000)  
        y = amplitude * np.sin(2 * np.pi * frequency * x + phase)

        return (x,y)

    def add_subtract_signals(self,signals,isAdd = True) :

        new_signal = {};

        for signal in signals :
            x,y = signal


            n = len(x)
            for i in range(len(x)) :
                if str(x[i]) in new_signal :
                    if isAdd :
                        new_signal[str(x[i])] += float(y[i])
                    else :
                        new_signal[str(x[i])] -= float(y[i])
                else :
                    new_signal[str(x[i])] = float(y[i])


        x = []
        y = []

        for k, v in new_signal.items():
            x.append(int(float(k)))
            y.append(v)

        return (x,y)

    def signal_scale(self,signal,scaler) :
        x,y = signal

        new_x = x.copy()
        new_y = y.copy()

        for i in range(len(new_x)) :
            new_y[i] = float(new_y[i]) * float(scaler)
        
        return (new_x,new_y)

    def signal_shift(self,signal,scaler) :
        x,y = signal

        new_x = x.copy()
        new_y = y.copy()

        for i in range(len(new_x)) :
            new_y[i] = float(new_y[i]) + float(scaler)
        
        return (new_x,new_y)

    def signal_square(self,signal) :
        x,y = signal

        new_x = x.copy()
        new_y = y.copy()

        for i in range(len(new_x)) :
            new_y[i] = float(new_y[i]) * float(new_y[i])
        
        return (new_x,new_y)

    def signal_commulation(self,signal) :
        x,y = signal

        new_x = x.copy()
        new_y = y.copy()

        for i in range(1,len(new_x)) :
            new_y[i] = float(new_y[i])
            new_y[i] += float(new_y[i-1])
        
        return (new_x,new_y)

    def normalize_signal(self, signal, new_min , new_max) :
        x,y = signal

        new_x = x.copy()
        new_y = y.copy()

        curr_max = max(new_y)
        curr_min = min(new_y)

        normalize_func = lambda _x : (((_x-curr_min)/(curr_max-curr_min))*(new_max-new_min))+new_min

        for i in range(len(new_y)) :
            new_y[i] = normalize_func(new_y[i]);
        
        print("========>",y,"\n",new_y)
        
        return (new_x,new_y)

    def loadTxtContent(self,path,isPolar = False) :
        try :
            data = open(path,'r').read().split('\n')

            isTimeDomain = bool(data[0]);
            isPeriodic = bool(data[1]);
            numberOfSamples = int(data[2])

            x = []
            y = []
            for i in range(3,len(data)):
                if not isPolar :
                    data[i] = data[i].split(' ')
                    try :
                        x.append(float(data[i][0]))
                        y.append(float(data[i][1]))
                    except:
                        print("Error in reading row",i)
                        continue
                else :
                    data[i] = data[i].split(',')
                    try :
                        x.append(float(data[i][0]))
                        y.append(float(data[i][1]))
                    except:
                        print("Error in reading row",i)
                        continue

            return (True,x,y)

        except :
            return (False,[],[])

class MyGUI(QMainWindow):

    #global data used across the class
    globalData = dict({})

    def __init__(self):
        super(MyGUI,self).__init__()
        uic.loadUi("mainPage.ui",self)

        #global data
        self.globalData = dict({
            "forPolarInfo" : [],
            "forImportedFiles" : [],
            "qntImportedFiles" : [],
            "opImportedFiles" : [],
            "dctImportedFiles" : [],
            "filterImportedFiles" : [],
            "opImportedWaves" : [],
            "ops" : {
                "multiply" : "multiply",
                "subtract" : "subtract",
                "square" : "square",
                "normalize" : "normalize",
                "comulate" : "comulate",
                "add" : "add",
                "shift" : "shift"
            },
            "getOpsRefs" : {
                "multiply" : lambda :  self.mulGroup,
                "shift" :  lambda :  self.mulGroup,
                "normalize" : lambda :  self.normGroup
            },
            "opsActions" : {
                "shift" : lambda : self.mulGroup.setVisible(bool(True)),
                "multiply" : lambda : self.mulGroup.setVisible(bool(True)),
                "normalize" : lambda : self.normGroup.setVisible(bool(True))
            },
            "funcs" : {
                
            },
            "test" : {}
        }) 

        #init functions
        self.show()
        self.setOpsVisibilityState(False)
        self.mulGroup.setVisible(bool(False))

        self.signalHelper = SignalHelper()
        self.customTest = CustomTest()

        #drop down options
        self.txtDis.triggered.connect(self.onTxtToDiscreteClick)
        self.txtCon.triggered.connect(self.onTxtToContinousClick)
        self.txtDisCon.triggered.connect(self.onTxtToDiscreteAndContinousClick)
        self.quatizationLevels.triggered.connect(self.qntLvlCustomTest)
        self.quantizationBits.triggered.connect(self.qntBitsCustomTest) 
        self.actiontxt_polar.triggered.connect(self.onImportPolar)
        #draw button
        self.drawBtn.clicked.connect(self.onDrawBtnClick)


        #Operations secion
        self.opImportBtn.clicked.connect(self.onOpImportBtn)
        self.opClrBtn.clicked.connect(self.onOpClrBtn)
        self.opDraw.clicked.connect(self.onOpDraw)
        self.opComboBox.currentIndexChanged.connect(self.onComboboxChanged)

        #quantization section
        self.qntBtn.clicked.connect(self.onQntBtnClicked)
        self.qntImportBtn.clicked.connect(self.onImportBtnClicked)

        #tranformation section
        self.forApplyBtn.clicked.connect(self.onForApplyBtn)
        self.forImportBtn.clicked.connect(self.onForImportBtnClicked)
        self.forSaveBtn.clicked.connect(self.onForSaveBtn)
        self.forApplyConstructBtn.clicked.connect(self.onForApplyConstructBtn)

        #DCT Section
        self.dctImportBtn.clicked.connect(self.onDctImportBtn)
        self.dctApplyBtn.clicked.connect(self.onDctApplyBtn)
        self.dctRemove.clicked.connect(self.onDctRemove)


        #filters Section
        self.filImp.clicked.connect(self.onFilImp)
        self.smoothApply.clicked.connect(self.onSmoothApply)
        self.dctRemove.clicked.connect(self.onDctRemove)
        self.firstAndSecDer.clicked.connect(self.onFirstAndSecDer)
        self.delayApply.clicked.connect(self.onDelayApply)
        self.foldApp.clicked.connect(self.onFoldApp)
        self.foldAppAndSave.clicked.connect(self.onFoldAppAndSave)
        self.RmvDCComp.clicked.connect(self.onRmvDCComp)

        #Convolution Section
        self.convolveBtn.clicked.connect(self.onConvolvePressed)
        self.convolveImpBtn.clicked.connect(self.onConvolveImpPressed)

        #Correlation
        self.correlateImpBtn.clicked.connect(self.onCorrelateImpPressed)


    def onCorrelateImpPressed(self):
        res= self._showOpenDialog()
        self.globalData["correlateImportedFiles"] = res
        signal_x= res[0][0]
        signal_y= res[0][1]
        kernel_x= res[1][0]
        kernel_y= res[1][1]
        SignalHelper.normalizedCrossCorrelation(res[0],res[1])

    def onConvolveImpPressed(self):
        res= self._showOpenDialog()
        self.globalData["convolveImportedFiles"] = res
        signal_x= res[0][0]
        signal_y= res[0][1]
        kernel_x= res[1][0]
        kernel_y= res[1][1]
        SignalHelper.convolve(signal_x,signal_y,kernel_x,kernel_y)




    def onConvolvePressed(self):
        print("apply")

    def onRmvDCComp(self) :
        signal = self.globalData["filterImportedFiles"][0]
        x,y = signal
        x = [*x]
        y = [*y]

        freq,amp,phase,polar = self.signalHelper.transformToFrequencyDomain(signal,1)
        self.globalData["forPolarInfo"] = polar
        print("polar : " , polar)

        polar[0][0] = 0
        polar[1][0] = 0 

        recSig = self.signalHelper.reconstructSignal(polar,len(signal))


        self.signalHelper.drawGraphs([(x,y),recSig],True,True,["Original","After"])

    def onFoldApp(self) :
        signal = self.globalData["filterImportedFiles"][0]
        sig = self.signalHelper.foldSignal(signal)
        self.signalHelper.drawGraphs([signal,sig],True,True,[ "Original","Delayed"])

    def onFoldAppAndSave(self) :
        signal = self.globalData["filterImportedFiles"][0]
        sig = self.signalHelper.foldSignal(signal)
        self.signalHelper.drawGraphs([signal,sig],True,True,[ "Original","Delayed"])
        self.globalData["filterImportedFiles"] = [sig]

    def onDelayApply(self):
        k = int(self.delayVal.text())
        signal = self.globalData["filterImportedFiles"][0]
        sig = self.signalHelper.delaySignal(signal,k)
        self.signalHelper.drawGraphs([signal,sig],True,True,[ "Original","Delayed"])

    def onFirstAndSecDer(self) :
        signal = self.globalData["filterImportedFiles"][0]
        x,y = signal
        x = [*x]
        y = [*y]
        sigs = self.signalHelper.computeFirstAndSecondDerivative(signal)
        self.signalHelper.drawGraphs([(x,y),*sigs],True,True,[ "Original","First Derivative" , "Second Derivative"])
        print("Conv Test case passed successfully")

    def onSmoothApply(self) :
        windowSize = int(self.smoothWin.text())
        signal = self.globalData["filterImportedFiles"][0]
        avgSignal = self.signalHelper.avgSignal(signal,windowSize)
        self.signalHelper.drawGraphs([signal,avgSignal],True,True,["Before", "After"])


    def onFilImp(self) :
        self.dctFileTxt.setPlainText("")
        res = self._showOpenDialog()
        self.filTxt.setPlainText (",".join(self.globalData["opImportedFiles"]))
        self.globalData["filterImportedFiles"] = res


    def onDctApplyBtn(self):
        m_coefficent = int(self.dct.text())
        signal = self.globalData["dctImportedFiles"][0]
        x_values = signal[0]
        y_values = signal[1]
        coeff= SignalHelper.compute_dct(x_values,y_values,m_coefficent)
        SignalHelper.save_dct_output(coeff)


    def onDctImportBtn(self) :
        self.dctFileTxt.setPlainText("")
        res = self._showOpenDialog()
        self.dctFileTxt.setPlainText (",".join(self.globalData["opImportedFiles"]))
        self.globalData["dctImportedFiles"] = res

    def onDctRemove(self):
        signal = self.globalData["dctImportedFiles"][0]
        x_values = signal[0]
        y_values = signal[1]
        SignalHelper.remove_DC(x_values, y_values)

    def onForApplyConstructBtn2(self):
        idxs = []
        idxs_map = {}
        newUpdates = str(self.textEdit.toPlainText()).split('\n')
        try:
            for i in range(len(newUpdates)) :
                newUpdates[i] = newUpdates[i].split(',')

                for j in range(len(newUpdates[i])) :
                    newUpdates[i][j] = float(newUpdates[i][j])
                
                idxs_map[newUpdates[i][0]] = [newUpdates[i][1],newUpdates[i][2]] 
                idxs.append(newUpdates[i][0])
        except :
            idxs = []
            self._showPopUp("Error" , "Error in parsin the changes string")
        


        
        print(newUpdates)
        print(idxs_map)

        signal = self.globalData["forImportedFiles"][0]
        freq,amp,phase,polar = self.signalHelper.transformToFrequencyDomain(signal,len(signal[0]))

        x_value = []
        for i in range(len(freq)) :
            x_value.append(i)
        waves = []
        for i in range(len(freq)) :
            _amp = 0
            phs = 0
            if (i in idxs) :
                phs = idxs_map[i][0]
                _amp = idxs_map[i][1]

            else :
                _amp = amp[i]
                phs = phase[i]

            res = self.signalHelper.buildWave(_amp,freq[i],phs,False,x_value)
            waves.append(res)
        
        

        _wave = self.signalHelper.add_subtract_signals(waves)
                
        self.signalHelper.drawGraphs([_wave])
        return
    
    def onForApplyConstructBtn(self):
        idxs = []
        idxs_map = {}
        newUpdates = str(self.textEdit.toPlainText()).split('\n')
        try:
            for i in range(len(newUpdates)) :
                newUpdates[i] = newUpdates[i].split(',')

                for j in range(len(newUpdates[i])) :
                    newUpdates[i][j] = float(newUpdates[i][j])
                
                idxs_map[int(newUpdates[i][0])] = [newUpdates[i][1],newUpdates[i][2]] 
                idxs.append(int(newUpdates[i][0]))
        except :
            idxs = []
            self._showPopUp("Error" , "Error in parsin the changes string")
        


        
        print(newUpdates)
        print(idxs_map)

        signal = self.globalData["forImportedFiles"][0]
        freq,amp,phase,polar = self.signalHelper.transformToFrequencyDomain(signal,len(signal[0]))

        imgs = []
        reals = []
        for i in range(len(freq)) :
            real = 0
            img = 0
            if (i in idxs) :
                phs = idxs_map[i][0]
                _amp = idxs_map[i][1]
                real = _amp / math.sqrt((math.tan(phs*math.pi/180))**2 + 1)*-1
                img = real * (math.tan(phs*math.pi/180))
            else :
                real = amp[i] / math.sqrt((math.tan(phase[i]*math.pi/180))**2 + 1)*-1
                img = real * (math.tan(phase[i]*math.pi/180))

                # res = self.buildWave(amp[i],freq[i],phase[i])

            imgs.append(img)
            reals.append(real)
        

        print("real-imgs\n",reals,'\n',imgs,'\n')
                
        wave = self.signalHelper.reconstructSignal((reals,imgs),len(freq))
        self.signalHelper.drawGraphs([wave])
        return

    def onImportPolar(self) :
        res = self._showOpenDialog(True)
        res = res[0]

        if (res == False) :
            return False; 

        print(res)
        signal = self.signalHelper.reconstructSignal(res,len(res))
        self.signalHelper.drawGraphs([signal])

    def onForSaveBtn(self) :
        polar = self.globalData["forPolarInfo"];
        if (polar == [] or len(polar) == 0) :
            self._showPopUp("Warning" , "No signals to save")
            return
        fileContent = "0\n0\n"+str(len(polar))+"\n"

        for i in range(len(polar)) :
            fileContent += str(polar[i][0]) + ',' + str(polar[i][1])
            if (i < len(polar)-1) :
                fileContent += '\n'
        
        currDate = str(date.today())
        file_path = "wave_" + currDate + ".txt"

        _file = open(file_path, "w")
        _file.write(fileContent)
        _file.close()

        self._showPopUp("Success" , "File saved succesfully, fileName: " + file_path)
    
        return

    def onForApplyBtn(self) :
        samplingFreq = int(self.forSamplingInp.text())
        signal = self.globalData["forImportedFiles"][0]
        freq,amp,phase,polar = self.signalHelper.transformToFrequencyDomain(signal,samplingFreq)
        self.globalData["forPolarInfo"] = polar

        print("freq : " , freq)
        print("amp : " , amp)
        print("phase : " , phase)
        print("polar : " , polar)

        self.signalHelper.drawGraphs([(freq,amp),(freq,phase)],True,False,["Frequnecy and Amplitude","Frequency and Phase"])

    def onForImportBtnClicked(self) :
        self.forFileTxt.setPlainText("")
        res = self._showOpenDialog()
        self.forFileTxt.setPlainText (",".join(self.globalData["opImportedFiles"]))
        self.globalData["forImportedFiles"] = res

    def qntBitsCustomTest(self) :
        print("File must be in the same directory")
        print(self.globalData["test"])

        res = self._showOpenDialog()
        fileName = self.globalData["opImportedFiles"][0]
        indexes = self.globalData["test"]["indexes"]
        binaries = self.globalData["test"]["binaries"]
        values = self.globalData["test"]["values"]
        errors = self.globalData["test"]["errors"]
        self.customTest.quatizationByLevel(fileName,indexes,binaries,values,errors,"1")

    def qntLvlCustomTest(self) :
        print("File must be in the same directory")
        print(self.globalData["test"])

        res = self._showOpenDialog()
        fileName = self.globalData["opImportedFiles"][0]
        indexes = self.globalData["test"]["indexes"]
        binaries = self.globalData["test"]["binaries"]
        values = self.globalData["test"]["values"]
        errors = self.globalData["test"]["errors"]
        self.customTest.quatizationByLevel(fileName,indexes,binaries,values,errors,"2")

    def onQntBtnClicked(self) :
        if (len(self.globalData["qntImportedFiles"]) == 0) :
            self._showPopUp("Warning" , "Please import files first")
            return
            
        isLvls = bool(self.qntLvlsOp.isChecked())
        isBits = bool(self.qntBitsOp.isChecked())
        levels  = float(self.qntLvls.text())
        quatizedValues = self.signalHelper.signalQuatization(self.globalData["qntImportedFiles"],levels)
        x = []
        y = []
        j = 0
        for i in quatizedValues:
            x.append(j)
            j += 1
            y.append(i["quantizedValue"])
        

        string = "";
        indexes = []
        binaries = []
        values = []
        errors = []
        if isLvls :
            length = math.ceil(math.log2(levels))
            string += "Index " + "Encode " + "Quantize " + "Error " + "\n"
            for i in quatizedValues :
                binStr = str("{0:b}".format((i["index"])))
                while(len(binStr) < length) :
                    binStr = '0'+binStr 

                indexes.append(int(i["index"])+1)
                values.append((i["quantizedValue"]))
                binaries.append(binStr)
                errors.append(i["error"])

                string += str(int(i["index"])+1) + " " + str(binStr) 
                string += " " + str(i["quantizedValue"]) + " " + str(i["error"]) + '\n'

        if isBits :
            string += "Encode " + "Quantize "+ "\n"
            length = math.ceil(math.log2(levels))
            for i in quatizedValues :
                binStr = str("{0:b}".format((i["index"])))
                while(len(binStr) < length) :
                    binStr = '0'+binStr 

                values.append((i["quantizedValue"]))
                binaries.append(binStr)
                string += str(binStr) + " " + str(i["quantizedValue"]) + '\n'

        self.globalData["test"] = {
            "indexes" : indexes,
            "values" : values,
            "binaries" : binaries,
            "errors" : errors
        }
        print(self.globalData["test"])
        self.qntOutTxt.setPlainText(string)
        self.signalHelper.drawGraphs([(x,y)],True,False)

    def onImportBtnClicked(self) :
        self.qntFileTxt.setPlainText("")
        res = self._showOpenDialog()
        self.qntFileTxt.setPlainText (",".join(self.globalData["opImportedFiles"]))
        self.globalData["qntImportedFiles"] = res
    
    def onComboboxChanged(self) :
        self.onOpClrBtn()
        self.setOpsVisibilityState(False)

        combo_box = self.findChild(QComboBox, "opComboBox")
        if combo_box is None:
            return

        choosenOp = str(combo_box.currentText()).strip().lower()

        if choosenOp in self.globalData["opsActions"] :
            self.globalData["opsActions"][choosenOp]()

    def onOpDraw(self) :
        if (len(self.globalData["opImportedWaves"]) == 0) :
            self._showPopUp("Warning" , "Please import files first")
            return
        
        ops = self.globalData["ops"]
        combo_box = self.findChild(QComboBox, "opComboBox")
        choosenOp = str(combo_box.currentText()).strip().lower()

        if choosenOp == ops["multiply"] : 
            scaler = float(self.opInpField.text())
            self.globalData["opImportedWaves"].append(self.signalHelper.signal_scale(self.globalData["opImportedWaves"][0],scaler))
        
        elif choosenOp == ops["add"] :
            self.globalData["opImportedWaves"].append(self.signalHelper.add_subtract_signals(self.globalData["opImportedWaves"]))

        elif choosenOp == ops["subtract"] :
            self.globalData["opImportedWaves"].append(self.signalHelper.add_subtract_signals(self.globalData["opImportedWaves"],False))

        elif choosenOp == ops["comulate"] :
            self.globalData["opImportedWaves"].append(self.signalHelper.signal_commulation(self.globalData["opImportedWaves"][0]))
        
        elif choosenOp == ops["square"] :
            self.globalData["opImportedWaves"].append(self.signalHelper.signal_square(self.globalData["opImportedWaves"][0]))

        elif choosenOp == ops["shift"] :
            scaler = float(self.opInpField.text())
            self.globalData["opImportedWaves"].append(self.signalHelper.signal_shift(self.globalData["opImportedWaves"][0],scaler))

        elif choosenOp == ops["normalize"] :
            _from = float(self.opInpFrom.text())
            _to   = float(self.opInpTo.text()) 

            if (_from > _to) :
                self._showPopUp("Warning" , "From can't be larger than To")
                return
            self.globalData["opImportedWaves"].append(self.signalHelper.normalize_signal(self.globalData["opImportedWaves"][0],_from,_to))

        


        
        self.signalHelper.drawGraphs(self.globalData["opImportedWaves"],True,True)

    def onOpClrBtn(self) :
        self.globalData["opImportedWaves"] = []
        self.globalData["opImportedFiles"] = []
        self.opFileNamesTxt.setPlainText("")

    def setOpsVisibilityState(self,state) :
        state = bool(state)

        for key in self.globalData["ops"]:
            if (key in self.globalData["getOpsRefs"]) :
                self.globalData["getOpsRefs"][str(key)]().setVisible(state)

    def onOpImportBtn(self) :
        res = self._showOpenDialog()
        self.opFileNamesTxt.setPlainText (",".join(self.globalData["opImportedFiles"]))
        self.globalData["opImportedWaves"] = res

    def _showPopUp(self,title,msgTxt) :
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(str(msgTxt))
        msg.setWindowTitle(str(title))
        result = msg.exec_()

    def onDrawBtnClick(self):

        analogFreq = float(self.anaInp.text())
        sampleFreq = float(self.sampInp.text())
        amplitude  = float(self.ampInp.text())
        phaseShift = float(self.phaseInp.text())
        isCos = bool(self.isCosBtn.isChecked())
        isSin = bool(self.isSinBtn.isChecked())

        if sampleFreq < analogFreq * 2 :
            self._showPopUp("Alias detected","The sampling frequency must be at least twice the signal frequency")
            return

        x,y = self.signalHelper.buildWave(float(amplitude),float( ((analogFreq))) ,float(phaseShift),float(isCos))
        
        
        self.signalHelper.drawGraphs([x],[y],True,True)

    def onTxtToDiscreteAndContinousClick(self):
        res = self._showOpenDialog()
        res = res[0]
        
        if (res == False) :
            return False;

        x,y = res
        self.signalHelper.drawGraphs( [(x,y)] ,True,True)
    
    def onTxtToDiscreteClick(self):
        res = self._showOpenDialog()
        res = res[0]

        if (res == False) :
            return False; 

        x,y = res
        self.signalHelper.drawGraphs([(x,y)],True,False)
    
    def onTxtToContinousClick(self):
        res = self._showOpenDialog()
        res = res[0]

        if (res == False) :
            return False;

        x,y = res
        self.signalHelper.drawGraphs([(x,y)][x],[y],False,True)

    def _showOpenDialog(self,isPolar = False):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.ExistingFiles  # Allow selecting multiple files

        fileDialog = QFileDialog(self)
        fileDialog.setFileMode(QFileDialog.ExistingFiles)
        fileDialog.setNameFilter("Text Files (*.txt)")

        if fileDialog.exec_():
            fileNames = fileDialog.selectedFiles()
            data = []

            for fileName in fileNames:
                state, x, y = self.signalHelper.loadTxtContent(fileName,isPolar)
                if state:
                    data.append((x, y))
                    self.globalData["opImportedFiles"].append(fileName)
                else:
                    self._showPopUp("Invalid format", f"Invalid input file format for {fileName}")
            return data
        else:
            self._showPopUp("Error", "An error occurred while importing files")
            return



def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()



if __name__ == '__main__':
    main()