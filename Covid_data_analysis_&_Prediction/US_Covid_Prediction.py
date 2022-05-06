# -*- coding: utf-8 -*-


import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
import os
import matplotlib.pyplot as plt
import datetime as dt


y = pd.read_excel('C:/Users/Downloads/US_dataset.xlsx')
dates = y['Date']
dates_list = [dt.datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').date() for date in dates]
y = y.drop(columns=['Country','Deaths', 'Recovered'])
y = y.groupby(['Date'])['Confirmed'].sum().reset_index()
y = y.drop(columns = ['Date'])
y = torch.FloatTensor(y['Confirmed'].values)
max_values = y.shape[0]
slice_index = int(0.7*max_values)
#%%Plot
def Subplot(arg1, arg2, filename, file_format):
    """
    plot a figure which have 2 subplots and save the figure to working directory
    
    parameters:
        arg1 : first variable to plot
        arg2 : second variable to plot
        filename : file name to be saved
        file_format : file format to be saved
        
    """
    
    plt.subplot(2,1,1)
    plt.ylabel("Loss")
    plt.plot(arg1)
    plt.title("training loss for US")
    plt.subplot(2,1,2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arg2)
    plt.tight_layout()
    plt.title("test loss for US")
    #plt.figure(figsize=(, 8))
    plt.savefig(filename +'.'+file_format,format=file_format)
    plt.close() 
    
def Pred_Plot(arg, filename, file_format, Title, Dates):
    arg = pd.DataFrame(arg).set_index([Dates])
    plt.plot(arg)
    plt.ylabel("Confirmed Cases")
    plt.xlabel("time")
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.title(Title)
    #plt.figure(figsize=(22, 8))
    plt.savefig(filename +'.'+file_format,format=file_format)
    plt.close()
    
def Pred_and_Original_Plot(arg1, arg2, filename, file_format, Title, Dates):
    arg1 = pd.DataFrame(arg1).set_index([Dates])
    arg2 = pd.DataFrame(arg2).set_index([Dates])
    plt.plot(arg1, 'b', label = 'Original')
    plt.plot(arg2, '-r', label = 'Predicted')
    plt.ylabel("Confirmed Cases")
    plt.xlabel("time")
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.title(Title)
    #plt.figure(figsize=(22, 8))
    plt.legend(loc="best")
    plt.savefig(filename +'.'+file_format,format=file_format)
    plt.close()

#%%dataset
class covid_dataset(torch.utils.data.Dataset):
    """
    split data for test and training purpose
    
    parameters:
        split_percentage:float,required
                         represents how much data for training
    returns:
        test and training data for original and noisy value
    
    """    
    def __init__(self, window_length, window_step, num_dim, split_percentage = 0.7, train = True, mean = None, std = None):
        self.split = split_percentage
        self.num_dim = num_dim
        self.train = train
        self.mean = mean
        self.std = std
        self.data = None
        self.label = None        
        y = pd.read_excel('C:/Users/tanvi/Downloads/US_dataset.xlsx')
        dates = y['Date']
        dates_list = [dt.datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').date() for date in dates]
        y = y.drop(columns=['Country','Deaths', 'Recovered'])
        y = y.groupby(['Date'])['Confirmed'].sum().reset_index()
        y = y.drop(columns = ['Date'])
        y = torch.FloatTensor(y['Confirmed'].values)
        max_values = y.shape[0]
        slice_index = int(self.split*max_values)

        
        if self.train:
            self.name = 'train_data'
            original_data = y[:slice_index]
            Pred_Plot(original_data, 'train_data', 'png', 'training data for US', dates_list[:slice_index])
            self.mean = original_data.mean(dim=0, keepdim=True)
            self.std = original_data.std(dim=0, keepdim=True)
            original_data = (original_data - self.mean) / self.std
            original_data = original_data.unfold(0, window_length, window_step)
            for i in range(original_data.shape[0]):
                    new_data = original_data[i, :20]
                    new_label = original_data[i, 20:]
                    if self.label is None and self.data is None:
                        self.data = new_data
                        self.label = new_label
                    else:
                        self.data = torch.vstack((self.data, new_data))
                        self.label = torch.vstack((self.label, new_label))
            self.data = torch.unsqueeze(self.data, dim = 1)
            
        else:
            self.name = 'test_data'
            original_data = y[slice_index:]
            Pred_Plot(original_data, 'test_data', 'png', 'test data for US', dates_list[slice_index:])
            original_data = (original_data - self.mean) / self.std
            original_data = original_data.unfold(0, window_length, window_step)
            for i in range(original_data.shape[0]):
                    new_data = original_data[i, :20]
                    new_label = original_data[i, 20:]
                    if self.label is None and self.data is None:
                        self.data = new_data
                        self.label = new_label
                    else:
                        self.data = torch.vstack((self.data, new_data))
                        self.label = torch.vstack((self.label, new_label))
            self.data = torch.unsqueeze(self.data, dim = 1)
            
            
    def __len__(self):
            return len(self.data)

    def __getitem__(self, index):
            x = self.data[index]
            y = self.label[index]
            return x, y

    
#%%dataloader
def dataloader(batch_size):
    
    """
    create dataloader where dataset is divided into batch which is required
    for training and testing of neural network
    
    parameter:
        train_dataset :dataset,required
        test_dataset  :dataset,required
        
    returns:
        dataloader for training and testing
    """
        
    
    train_dataset = covid_dataset(window_length = 30, window_step = 5, num_dim = 1)
    test_dataset = covid_dataset(window_length = 30, window_step = 5, num_dim = 1, train = False, mean = train_dataset.mean,
                                     std = train_dataset.std)
        

        # load data sets and create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size = batch_size, shuffle = False, pin_memory= True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                batch_size = batch_size, shuffle = False, pin_memory = True)
    return train_loader, test_loader


#%%Forecasting model

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model,self).__init__()
        self.c1 = nn.Conv1d(1, 32, kernel_size = 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv1d(32, 64, kernel_size = 3)
        self.relu2 = nn.ReLU(inplace=True)
        self.f1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64*16, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.relu2(x)
        x = self.f1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class Train_Test(nn.Module):
    
    def __init__(self, model,criterion, lr):
        super(Train_Test, self).__init__()        
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.test_loss = 0
        self.print_results = True
        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)
        
    
    def Train(self, train_data, epoch): 
        """
        training autoencoder
        
        parameters:
            train_dataloader:dataloader,required
            model : required
            
        return:
            train_error
        
        """
        self.model.train()    
        train_error = 0
        n_iteration = len(train_data)
        pred_tr = []
        for i, (inputs, target) in enumerate(train_data):        
            target = target.to(device)
            inputs = inputs.to(device)
            self.optimizer.zero_grad()              
            pred = self.model(inputs)
            pred_tr.append(pred)
            loss = torch.sqrt(self.criterion(pred, target))
            loss.backward()
            self.optimizer.step()
            train_error += loss.item()
        pred_tr = torch.cat(pred_tr,dim=0)
        train_loss =  train_error/n_iteration
        return train_loss, pred_tr           

    def test(self, test_data,epoch):
        """
        testing autoencoder
        
        parameters:
            test_dataloader: dataloader,required
            model : required
        return:
            test_error
        
        """
        self.model.eval()
        with torch.no_grad():
            test_error = 0
            test_iteration = len(test_data)
            pred = []
            test_data_store = []
            for i,(inputs, target) in enumerate(test_data):
                target = target.to(device)
                inputs = inputs.to(device)
                pred_t = self.model(inputs)                
                pred.append(pred_t)
                test_data_store.append(inputs)
                #test_label.append(label)
                tst_loss = torch.sqrt(self.criterion(pred_t, target))
                test_error+= tst_loss.item()       
            pred = torch.cat(pred, dim = 0)
            test_data_store = torch.cat(test_data_store, dim = 0)
            test_loss = test_error/test_iteration            
        return test_loss, pred, self.model.state_dict(), test_data_store 
        
        
def objective(trial):
    
    epochs = 500
    window_length = 30
    window_step = 5
    num_dim = 1
    best_loss = 9999
    #best_train_loss = 99999
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    #kernel = trial.suggest_int('kernel', 2, 6, log = True)
    batch_size = trial.suggest_int('batch_size', 8, 16, log = True)    
    train_data, test_data = dataloader(batch_size)
    
    model = CNN_Model().to(device)  
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    train_test_loop = Train_Test(model, nn.MSELoss(), lr)
    #test_error = train_test_loop(model,train_dataloader, test_dataloader,n_epochs,number_of_signal)
    test_losses = []
    test_pred = []
    train_losses = []
    predicted_value = torch.tensor([])
    
                                                                                                          
    for epoch in range(0, epochs):
        train_loss, pred_tr = train_test_loop.Train(train_data, epoch)
        train_losses.append(train_loss)        
        test_loss, pred, model_state_dict, test_data_store = train_test_loop.test(test_data, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            #best_pred = pred
            #best_epoch = epoch
            torch.save(model_state_dict, os.path.join('model.pt'))
        test_pred.append(pred)
        test_losses.append(test_loss)
        trial.report(test_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
                                                                 
  
    #return test_losses
    Subplot(train_losses, test_losses, 'train_test_loss', 'png')
    min_loss = min(test_losses)
    min_loss_index = test_losses.index(min_loss)
    #min_loss_epoch = (min_loss_index+1)
    min_loss_pred = test_pred[min_loss_index]
    test_data_store = torch.squeeze(test_data_store, dim = 1)
    test_data_final = torch.cat((test_data_store, min_loss_pred), dim = 1)
    #print(test_data_final.shape)
    data = covid_dataset(window_length, window_step, num_dim)
    mean = data.mean
    std = data.std
    #print(std.shape, mean.shape)
    for i in range(test_data_final.shape[0]):    
        if i  == 0:
            predicted_value = test_data_final[0,:] 
        else:
            predicted_value = torch.cat((predicted_value, test_data_final[i, 25:]))
    predict = predicted_value * std + mean
    original_test = y[slice_index:]
    Pred_and_Original_Plot(original_test[3:], predict, 'prediction', 'png','prediction plot for France', dates_list[-predict.shape[0]:])
    return test_losses[-1]



# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda:0')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')  
    
    
#%%

if __name__ =="__main__":
    
    #data = torch.double()
       
    study = optuna.create_study(direction = "minimize")
    study.optimize(objective, n_trials = 200)
    study.trials_dataframe()
    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
