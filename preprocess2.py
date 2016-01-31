#!/usr/bin/python

import sys
import csv
import random

def writeCSV(inputFile, outputFile, answerFile):
    # name
    # output
    # input 1
    # ...
    # input 46    

    # cw.writerow(["Name","Address","Telephone","Fax","E-mail","Others"])

    cr = csv.reader(open(inputFile,"rb"))
    cw = csv.writer(open(outputFile, "wb"))
    answer = open(answerFile, "w")

    count = 0
    
    
    # find min, max
    track = [[0 for x in range(2)] for x in range(46)] 
    for row in cr:
        if (count != 0):
            for i in range(2,48):
                if float(row[i]) < track[i-2][0]:
                    track[i-2][0] = float(row[i])
                if float(row[i]) > track[i-2][1]:
                    track[i-2][1] = float(row[i])
        count = count + 1
    
    print 'sosad'
    print track
    
    cr = csv.reader(open(inputFile,"rb"))
    count = 0
    # scaled_feature = (feature - min(featurearray))/(max(featurearray) - min(featurearray)).
    for row in cr:
        #print row
        if (count != 0):
            entry = []
            #for i in range(2,48):
            for i in range(2,25):
                #print row[i]
                value = (float(row[i]) - track[i-2][0]) / (track[i-2][1] - track[i-2][0])
                #print value
                entry.append(value)

            if (float(row[1])<6.0):
                entry.append("1")
                entry.append("0")
                answer.write('0\n')
            elif (float(row[1])>=6.0):
                entry.append("0")
                entry.append("1")
                answer.write('1\n')
            else:
                entry.append("0")
                entry.append("0")
                answer.write('?\n')
            cw.writerow(entry)
    	count = count + 1
    answer.close()
    

def readCSV():
    cr = csv.reader(open("data.csv","rb"))
    training_sets=[]

    count = 0
    for row in cr:
        if count != 0:    
	           # print row
	            	        # print index 2, second to last entry
	        # print row[2], row[-2]	
	        whole = []
	        input = []
	        output = []
	        for i in range(0,46):
	            input.append(float(row[i]))
	        
	        output.append(float(row[46]))
	        whole.append(input)
	        whole.append(output)
	        training_sets.append(whole)
       	count = count + 1
    
    return training_sets



if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    answerFile = sys.argv[3]
    writeCSV(inputFile, outputFile, answerFile)