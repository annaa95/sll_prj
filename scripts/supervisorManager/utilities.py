import numpy as np
import scipy.signal as sig

def lowPassfilter(signal, opts):
    # Create a FIR filter and apply it to x.
    #------------------------------------------------
    # The Nyquist rate of the signal.
    nyq_rate = opts["Fs"] / 2.0
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = opts["width"]/nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = opts["F_att"]
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = sig.kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    cutoff_hz = opts["F_cut"]
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = sig.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    # Use lfilter to filter x with the FIR filter.
    filtered_x = sig.lfilter(taps, 1.0, signal)

    return filtered_x[-1]



def editWBO(filename):
    """
    this function modifies the translation and rotation fileds before inserting the robot filed in the tree structure
    filename type: string
    translation is @index 2
    rotation is @index 3
    """
    a = []
    b = []
    my_file = open(filename)
    string_list = my_file.readlines()
    my_file.close()
    translation = string_list[2][14:-1]
    incipit_t = string_list[2][0:14]
    closure = string_list[2][-1]
    rotation = string_list[3][11:-1]
    incipit_r = string_list[3][0:11]
    t = ['0.0', '0.6', '0.0']
    r = ['0.0', '0.0', '1.0', '0.0']
    for i in range(len(t)):
        try:
            a.append(float(t[i])+(0.1*np.random.rand()-0.05))
        except:
            pass
    # b ha 4 elementi
    for i in range(len(r)):
        try:
            b.append(float(r[i])+(0.1*np.random.rand()-0.05))
        except:
            pass
    # list to string
    stringa= ' '.join([str(elem) for elem in a]) 
    stringb= ' '.join([str(elem) for elem in b])
    list1 = [incipit_t,stringa, closure]
    list2 = [incipit_r, stringb, closure]
    insert_t = ' '.join(list1)
    insert_r = ' '.join(list2)
    string_list[2] = insert_t
    string_list[3] = insert_r
    my_file = open(filename, 'w')
    new_file_contents = "".join(string_list)
    my_file.write(new_file_contents)
    my_file.close()
    readable_file = open(filename)
    read_file = readable_file.read()


def normalizeToRange(value, minVal, maxVal, newMin, newMax, clip=False):
    """
    Normalize value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :param minVal: value's min value, value ∈ [minVal, maxVal]
    :param maxVal: value's max value, value ∈ [minVal, maxVal]
    :param newMin: normalized range min value
    :param newMax: normalized range max value
    :param clip: whether to clip normalized value to new range or not
    :return: normalized value ∈ [newMin, newMax]
    """
    value = float(value)
    minVal = float(minVal)
    maxVal = float(maxVal)
    newMin = float(newMin)
    newMax = float(newMax)

    if clip:
        return np.clip((newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax, newMin, newMax)
    else:
        return (newMax - newMin) / (maxVal - minVal) * (value - maxVal) + newMax
        
class real_time_peak_detection():

    def __init__(self, array, lag, threshold, influence):
        self.y = list(array) # input min length lag+2
        self.length = len(self.y)
        # algorithm parameters
        self.lag = lag
        self.threshold = threshold
        self.influence = influence

        self.signals = [0] * len(self.y) # Initialize signal results
        self.filteredY = np.array(self.y).tolist() # Initialize filtered series
        
        self.avgFilter = [0] * len(self.y) # Initialize average filter
        self.stdFilter = [0] * len(self.y) # Initialize standard deviation filter
        self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
        self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

    def thresholding_algo(self, new_value):
        self.y.append(new_value)
        i = len(self.y) - 1
        self.length = len(self.y)
        if i < self.lag:
            return 0
        elif i == self.lag:
            self.signals = [0] * len(self.y)
            self.filteredY = np.array(self.y).tolist()
            self.avgFilter = [0] * len(self.y)
            self.stdFilter = [0] * len(self.y)
            self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
            self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
            return 0

        self.signals += [0]
        self.filteredY += [0]
        self.avgFilter += [0]
        self.stdFilter += [0]

        if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
            if self.y[i] > self.avgFilter[i - 1]:
                self.signals[i] = 1
            else:
                self.signals[i] = -1

            self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
        else:
            self.signals[i] = 0
            self.filteredY[i] = self.y[i]
            self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
            self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

        return self.signals[i]
