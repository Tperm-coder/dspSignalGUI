import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np

class SignalHelper:
    def __init__(self):
        pass

    def drawGraphs(self, graphs, isDiscrete=True, isContinuous=True):
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

        plt.tight_layout()
        plt.show()

    def buildWave(self, amplitude=1.0, frequency=1.0, phase=0.0 , isCos = False):
        if isCos : 
            phase += 90.0

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

    def quantize_signal(self,signal, num, levels=True):
        num_levels = 1
        if levels:
            num_levels = int(num)
        else:
            num_levels = pow(2, int(num))

        x_min,x_max = min(signal[0][0]), max(signal[0][0])
        y_min , y_max = min(signal[0][1]),max(signal[0][1])
        x_step_size = (x_max - x_min) / (num_levels - 1)
        y_step_size = (y_max - y_min) / (num_levels - 1)

        quant_x=[]
        for x in signal[0][0]:
            quant_x.append(((x-x_min)/x_step_size)*x_step_size+x_min)
        quant_y = []
        for y in signal[0][1]:
            quant_y.append(((y-y_min)/y_step_size)*y_step_size+y_min)

        error_x = []
        for i in range(len(signal[0][0])):
            error_x.append(signal[0][0][i] - quant_x[i])
        error_y = []
        for i in range(len(signal[0][1])):
            error_y.append(signal[0][1][i] - quant_x[i])
        tuple=(quant_x,quant_y)
        arr = []
        arr.append(signal[0])
        arr.append(tuple)
        # ----------------------------- Ekteb Hena ----------------------------
        self.drawGraphs(arr)
        return (quant_x,quant_y)

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

    #global data used across the class
    globalData = dict({})

    def __init__(self):
        super(MyGUI,self).__init__()
        uic.loadUi("mainPage.ui",self)
        self.quantData = 0
        #global data
        self.globalData = dict({
            "opImportedFiles" : [],
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
                
            }
        }) 

        #init functions
        self.show()
        self.setOpsVisibilityState(False)

        #class used to perform signal operations and logic
        self.signalHelper = SignalHelper()
        self.mulGroup.setVisible(bool(False))

        #drop down options
        self.txtDis.triggered.connect(self.onTxtToDiscreteClick)
        self.txtCon.triggered.connect(self.onTxtToContinousClick)
        self.txtDisCon.triggered.connect(self.onTxtToDiscreteAndContinousClick)

        #draw button
        self.drawBtn.clicked.connect(self.onDrawBtnClick)


        #Operations secion
        self.opImportBtn.clicked.connect(self.onOpImportBtn)
        self.opClrBtn.clicked.connect(self.onOpClrBtn)
        self.opDraw.clicked.connect(self.onOpDraw)

        self.opComboBox.currentIndexChanged.connect(self.onComboboxChanged)

        # Quantization Seciton
        self.opImportBtn_2.clicked.connect(self.onQuantImportBtn)
        self.quantizeBtn.clicked.connect(self.onQuantizeBtn)
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

    def onQuantImportBtn(self) :
        res = self._showOpenDialog()
        self.quantData = res

    def onQuantizeBtn(self):
        text = self.quantize_field.text()
        levels = self.isLevel.isChecked()
        self.signalHelper.quantize_signal(self.quantData,text,levels)


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
                    self.globalData["opImportedFiles"].append(fileName.split('/').pop())
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