import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np

class SignalHelper:
    def __init__(self):
        pass

    def drawGraph(self,x,y,isDiscrete = True ,isContinous = True):
        plt.show()

        x = np.array(x)
        y = np.array(y)

        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = np.interp(x_smooth, x, y)

        if isDiscrete :
            plt.plot(x, y, 'ro', label='Data Points', linestyle='none') 

        if isContinous :
            plt.plot(x_smooth, y_smooth, 'b-')

        plt.legend()
        plt.show()

    def buildWave(self, amplitude=1.0, frequency=1.0, phase=0.0 , isCos = False):
        if isCos : 
            phase += 90.0

        x = np.linspace(0, 2 * np.pi, 1000)  
        print(x)
        print(frequency)
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
                        new_signal[str(x[i])] += y[i]
                    else :
                        new_signal[str(x[i])] -= y[i]
                else :
                    new_signal[str(x[i])] = y[i]


        x = []
        y = []

        for k, v in new_signal.items():
            x.append(int(k))
            y.append(v)

        return (x,y)

    def signal_commulation(self,signal) :
        x,y = signal

        for i in range(1,len(x)) :
            y[i] += y[i-1]
        
        return (x,y)

    def normalize_signal(self, signal, new_min , new_max) :
        x,y = signal

        curr_max = max(y)
        curr_min = min(y)

        normalize_func = lambda x : (((x-curr_min)/(curr_max-curr_min))*(new_max-new_min))+new_min

        for i in range(len(y)) :
            y[i] = normalize_func(y[i]);
        
        return (x,y)


    def loadTxtContent(self,path) :
        try :
            data = open(path,'r').read().split('\n')

            isTimeDomain = bool(data[0]);
            isPeriodic = bool(data[1]);
            numberOfSamples = int(data[2])

            x = []
            y = []
            for i in range(3,len(data)):
                data[i] = data[i].split(' ')
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

    def __init__(self):
        super(MyGUI,self).__init__()
        uic.loadUi("mainPage.ui",self)
        self.show()

        self.signalHelper = SignalHelper()

        #drop down options
        self.txtDis.triggered.connect(self.onTxtToDiscreteClick)
        self.txtCon.triggered.connect(self.onTxtToContinousClick)
        self.txtDisCon.triggered.connect(self.onTxtToDiscreteAndContinousClick)

        #draw button
        self.drawBtn.clicked.connect(self.onDrawBtnClick)

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
        
        
        self.signalHelper.drawGraph(x,y,True,True)


    def onTxtToDiscreteAndContinousClick(self):
        res = self._showOpenDialog()
        
        if (res == False) :
            return False;

        x,y = res
        self.signalHelper.drawGraph(x,y,True,True)
    
    def onTxtToDiscreteClick(self):
        res = self._showOpenDialog()
        if (res == False) :
            return False; 

        x,y = res
        self.signalHelper.drawGraph(x,y,True,False)
    
    def onTxtToContinousClick(self):
        res = self._showOpenDialog()
        if (res == False) :
            return False;

        x,y = res
        self.signalHelper.drawGraph(x,y,False,True)


    def _showOpenDialog(self):
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
                state, x, y = self.signalHelper.loadTxtContent(fileName)
                if state:
                    data.append((x, y))
                else:
                    self.showPopUp("Invalid format", f"Invalid input file format for {fileName}")

            return data
        else:
            return None





def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()



if __name__ == '__main__':
    main()