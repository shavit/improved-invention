class Dog(object):

    def __init__(self, name, temperature):
        self.__name = name
        self.__temperature = temperature

    def status(self):
        print('[Dog] Name: {}, Temperature: {}'.format(self.__name, self.__temperature))

    def setTemperature(self, temperature):
        self.__temperature = temperature
