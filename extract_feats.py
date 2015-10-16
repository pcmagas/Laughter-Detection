import xml.etree.cElementTree as ET
from scikits.audiolab import wavread, play
import pickle
from scikits.talkbox.features import mfcc
from glob import glob
import os
import numpy as np
from sklearn.externals import joblib

def get_slices(data, time, frequency, overlay = 0):
    n_samples = time*frequency
    cnt = 0
    slices = []
    while (cnt+n_samples < len(data)):
        slices.append(data[cnt:cnt+n_samples])
        cnt = cnt + (1-overlay) * n_samples

    return slices


#Collect the wav files and their transcriptions
xml_files= glob("./ami_public_manual/words/EN*.xml")
xml_files.sort()

wav_files = [y for x in os.walk('./amicorpus') for y in glob(os.path.join(x[0], 'EN*Headset-*.wav'))]

wav_files.sort()


#Retrieve positive data and their features
print 'Retrieve positive data and their features'
train_data = []
X = []
for i in range(0,len(xml_files)):
    tree = ET.ElementTree(file=xml_files[i])
    data, sample_frequency,encoding = wavread(wav_files[i])
    print "**"
    print xml_files[i]
    print wav_files[i]
    for elem in tree.iterfind('vocalsound[@type="laugh"]'):
        if (float(elem.attrib['endtime']) - float(elem.attrib['starttime']) < 0.5):
            continue
        #print elem.tag, elem.attrib
        #play(data[float(elem.attrib['starttime'])*sample_frequency: float(elem.attrib['endtime'])*sample_frequency], sample_frequency)
        for data_slices in get_slices(data[float(elem.attrib['starttime'])*sample_frequency: float(elem.attrib['endtime'])*sample_frequency], 1, sample_frequency):
            train_data =  get_slices(data_slices, 0.025, sample_frequency, 0.2)

            mfcc_list = []
            for d in train_data:
                if (d != []):
                    ceps,mspec, spec = mfcc(d)
                    mfcc_list.append(ceps[0])
            #Calculate the mean and variance of the mfcc features
            mean = []
            variance = []
            if (len(mfcc_list) > 0):
                for i in range(0,len(mfcc_list[0])):
                    mean.append(np.mean(mfcc_list[:][i]))
                    variance.append(np.var(mfcc_list[:][i]))
                X.append(mean+variance)
                    

print "Saving to disk"
joblib.dump(X,"models/train_data_pos_mfcc.pkl")

#Retrieve negative data and their features
print "Retrieving negative data and their features"
train_data = []
X = []
for i in range(0,len(xml_files)):
    tree = ET.ElementTree(file=xml_files[i])
    data, sample_frequency,encoding = wavread(wav_files[i])
    print "--"
    print xml_files[i]
    print wav_files[i]
    for elem in tree.iter():
        if (elem.tag != 'w'):
            continue
        if (float(elem.attrib['endtime']) - float(elem.attrib['starttime']) < 0.5):
            continue
        #print elem.tag, elem.attrib
        #play(data[float(elem.attrib['starttime'])*sample_frequency: float(elem.attrib['endtime'])*sample_frequency], sample_frequency)
        for data_slices in get_slices(data[float(elem.attrib['starttime'])*sample_frequency: float(elem.attrib['endtime'])*sample_frequency], 1, sample_frequency):
            train_data =  get_slices(data_slices, 0.025, sample_frequency, 0.2)

            mfcc_list = []
            for d in train_data:
                if (not np.all(d==0)):
                    try:
                        ceps,mspec, spec = mfcc(d)
                    except Warning:
                        continue
                    else:
                        if (not np.isnan(ceps[0]).any()):
                            mfcc_list.append(ceps[0])

            #Calculate the mean and variance of the mfcc features
            mean = []
            variance = []
            if (len(mfcc_list) > 0):
                for i in range(0,len(mfcc_list[0])):
                    mean.append(np.mean(mfcc_list[:][i]))
                    variance.append(np.var(mfcc_list[:][i]))
                X.append(mean+variance)


print "Saving to disk"
joblib.dump(X,"models/train_data_neg_mfcc.pkl")
