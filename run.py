from farm.network import Network
from numpy import arange

if __name__ == '__main__':
    data = arange(0.3 ,1.3, 0.001)
    target = arange(0.6 ,1.6, 0.001)

    iNodes = len(data)
    hNodes = len(data)
    oNodes = len(data)
    rate = 0.001
    network = Network(iNodes, hNodes, oNodes, rate)
    network.train(data, target)
    res = network.query(arange(0.0 ,1.0, 0.001))

    print('[Network] Out:', min(res), max(res))
