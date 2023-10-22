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
        self._showOpenDialog(True,True)
    
    def onTxtToDiscreteClick(self):
        self._showOpenDialog(True,False)
    
    def onTxtToContinousClick(self):
        self._showOpenDialog(False,True)
 
    def _showOpenDialog(self,isDiscrete = True ,isContinous = True):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt)", options=options)

        if not fileName:
            self.showPopUp("Invalid format","Invalid input file format")
            return False

        state,x,y = self.signalHelper.loadTxtContent(fileName)

        if (not state) :
            return False
        
        self.signalHelper.drawGraph(x,y,isDiscrete,isContinous)
  



def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()



if __name__ == '__main__':
    main()