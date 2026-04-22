import psutil
import os


class findSDcardAdd():
    
    def findMount(self):
        for prt in psutil.disk_partitions():
            if "vfat"in prt.fstype or "sd" in prt.device:
                return prt.mountpoint
        return None
    