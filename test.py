import math
def reconstructSignal(signalPolar,samplingFreq) :
        N = len(signalPolar)
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

                valReal = signalPolar[_n][0]
                valImg = signalPolar[_n][1]


                if (curReal > 0 or curReal < 0) :
                    # el imaginary hayteer wel real hayscal
                    _sum += valReal*curReal
                else :
                    # el imaginary hayb2a real wel real hayteer
                    _sum += valImg*curImg*-1 # -1 34an el img yb2a real
                    
            signal.append(_sum/samplingFreq)
        
        return signal


print(reconstructSignal([(6,0),(-2,2),(-2,0),(-2,-2)],4))