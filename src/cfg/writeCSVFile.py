import csv
import os


class writeToCSV():

    def __init__(self, savePath):
        self.savePath = savePath
        self.file = None
        self.writer = None


    def openFile(self):
        self.file = open(self.savePath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frameNumber", "License-Plate"])

    def writeData(self, frameCounter, lp, imgpath):
        nlist = [frameCounter, lp, imgpath]
        self.writer.writerow(nlist)
        self.file.flush()
        os.fsync(self.file.fileno())
        
    def closeFile(self):
        self.file.close()
        self.file = None
        self.writer = None