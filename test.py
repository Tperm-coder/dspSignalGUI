import matplotlib.pyplot as plt


def loadTxtContent(path) :
    data = open(path,'r').read().split('\n')

    try :
        for i in range(len(data)):
            data[i] = float(data[i])
    except:
        return (False,data)

    return (True,data)



def drawGraph():
    state,y = getRecords("./test.txt")

    if (not state) :
        return False

    x = []
    for i in range(0,len(y)) :
        x.append(i)
        
    print(y)

    plt.plot(x, y, linestyle='-')
    plt.show()



print(drawGraph())