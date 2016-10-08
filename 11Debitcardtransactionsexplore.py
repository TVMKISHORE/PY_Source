import pandas as pd
user_cols=['date','Narration','valdate','debamount','credamount','checkref','clobal']
Debitstat = pd.read_table('Debitstat.csv', sep=',',header=0,names=user_cols)
Debitstat['debamount'] = Debitstat.debamount.astype('float')
Debitstat['credamount'] = Debitstat.credamount.astype('float')
Debitstat.debamount.sum()
Debitstat.credamount.sum()
Debitstat['date'] = pd.to_datetime(Debitstat.date)
Debitstat['valdate'] = pd.to_datetime(Debitstat.valdate)
import matplotlib.pyplot as plt

#**************************************
#**************************************

Debitstat.debamount.plot(kind='bar')
Debitstat.credamount.plot(kind='bar')
#-----------------------------------
Debitstat.Narration.value_counts()
#-----------------------------------

np.arange(0,Debitstat.shape[0])
Y=np.arange(0,Debitstat.shape[0])
Debitstat['colnum']= Y
Debitstat.plot(kind='scatter',y='debamount',x='colnum')
Debitstat.plot(kind='scatter',y='credamount',x='colnum')

#-------------------------------------------
HCL salaty plots
#-------------------------------------------
Debitstat.query('credamount >50000').plot(kind='scatter',y='credamount',x='colnum')
Debitstat.query('credamount >50000').plot(kind='bar',y='credamount',x='colnum')
#---------------------------------------------
