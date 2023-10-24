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

    
x = [
    ([0,1,2,3],[10,10,10,10]),
    ([0,10,20],[10,10,10])
]  

print(normalize([10,20,30,40,50],1,5))