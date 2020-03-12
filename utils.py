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
        lns11 = ax1.plot(loss_x, loss_y, 'b-', label='train loss')
        lns12 = ax1.plot(self.valid_epoch, self.valid_loss, 'c-', label='train loss')
        lns21 = ax2.plot(accs_x, accs_y, 'g-', label='accuracy')
        lns22 = ax2.plot(errs_x, errs_y, 'r-', label='error rate')

        lns = lns11 + lns12 + lns21 + lns22
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=7)

        ax1.set_xlabel("epoch index")
        ax1.set_ylabel("loss", color='b')
        ax2.set_ylabel("acc/error")

        plt.savefig(self.fname,dpi=500)
        plt.show()