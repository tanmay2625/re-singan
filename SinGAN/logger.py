import os
import datetime



class logger:
    def __init__(self) -> None:
        pass

    @classmethod
    def initiate(self,opt):
        dirName="logs"
        try:
            os.makedir(dirName)
        except:
            pass
        x=datetime.datetime.today()
        self.logFileName= 'log_alpha=%d_%s_time=%s.txt'%(opt.alpha,opt.training_name, x.strftime("%b-%d-%Y-%H:%M:%S"))
        self.file= open(('%s/%s')%(dirName,self.logFileName),'x')
    
    @classmethod
    def log_(self,s):
        self.file.writelines(s)
        self.file.writelines('\n')
    
    @classmethod
    def close_(self):
        self.file.close()

