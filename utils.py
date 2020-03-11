import matplotlib.pyplot as plt
class Plot():
    def __init__(self,nepoch,fname='data/loss.png'):
        self.loss=[]
        self.valid_loss=[]
        self.valid_epoch=[]
        self.accs=[]
        self.errs = []
        self.accs_index = []
        self.nepoch=nepoch
        self.fname=fname
        pass

    def add_loss(self,loss):
        self.loss.append(loss)

    def add_acc(self,acc,epoch):
        self.accs.append(acc)
        self.errs.append(1-acc)
        self.accs_index.append(epoch)

    def add_valid_loss(self,loss,epoch):
        self.valid_loss.append(loss)
        self.valid_epoch.append(epoch)


    def show(self):
        loss_x=[ (i+1)*(self.nepoch/len(self.loss)) for i in range(len(self.loss))]
        loss_y=self.loss

        accs_x=self.accs_index
        accs_y=self.accs

        errs_x = self.accs_index
        errs_y = self.errs

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(loss_x, loss_y, 'b-', label='train loss')
        ax1.plot(self.valid_epoch, self.valid_loss, 'c-', label='train loss')
        ax2.plot(accs_x, accs_y, 'g-', label='accuracy')
        ax2.plot(errs_x, errs_y, 'r-', label='error rate')
        ax1.set_xlabel("epoch index")
        ax1.set_ylabel("loss", color='b')
        ax2.set_ylabel("accuracy/error")
        fig.legend(loc=9)
        # plt.plot(loss_x,self.loss)
        plt.savefig(self.fname,dpi=500)
        plt.show()