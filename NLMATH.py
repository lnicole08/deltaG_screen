# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:22:10 2023

@author: lnico
"""

def frames(df):
   
    time = df['Seconds'].iloc[2] - df['Seconds'].iloc[1]
    if time == 1.0:
        fps =1
    if time <1.0:
        fps =5
    
    return fps

def calcgraph(df, filterword):
    import pandas as pd
  
    phase = ["Dark", "Full", "Recovery"]
    df4 = pd.DataFrame()
    for n in phase:        
        df_sd = df[(df["ExperimentState"] == "Assimilation time - "+n)|(df["ExperimentState"] == n)].reset_index(drop=True)
        df_time = pd.DataFrame()
        if n == "Full":
            df_time['Seconds'] = df_sd['Seconds']-23
        if n == "Dark":
            df_time['Seconds'] = df_sd['Seconds']
        if n == "Recovery":
            df_time['Seconds'] = df_sd['Seconds']-46
            
        df_sd2 = df_sd.filter(regex=filterword)
        df_time["ExperimentState"] = n
        df_sd2 = abs(df_sd2)
        df_sd3 = pd.concat([df_time, df_sd2], axis=1)
        df4 = pd.concat([df4, df_sd3])
        
    return df4

def meangraph(df):
    import pandas as pd
    phase = ['Dark','Full', 'Recovery']
    dft = pd.DataFrame()
    for n in phase:
        df1 = df[(df["ExperimentState"] == "Assimilation time - "+n)|(df["ExperimentState"] == n)].reset_index(drop=True)
        df_meand = pd.DataFrame()
        df_meand['Seconds']=df1['Seconds']
        df_meand['ExperimentState'] = n
        df_meand['mean']= df1.iloc[:,2:].mean(axis=1)
        df_meand['CI']= df1.iloc[:,2:].sem(axis=1)*1.96        
        dft = pd.concat([dft, df_meand])
    return dft

def fallcalc(df, phase):
    import pandas as pd
    
    dff = df.filter(regex="Fall.*")
    dff = pd.concat([df.iloc[:,0:2], dff], axis=1)
    dff2 = dff[(dff["ExperimentState"] == "Assimilation time - "+phase)|(dff["ExperimentState"] == phase)].reset_index(drop=True)
    nnumber = len(dff2.iloc[:,2:].columns)
    dff2["Total falls per sec"] = (dff2.iloc[:,2:].sum(axis=1))/nnumber
    
    return dff2

def violinfall(df, phase): #obsolete
    import pandas as pd
    import numpy as np
    
    dffall = df.filter(regex="Fall.*")
    dffall = pd.concat([df.iloc[:,0:2], dffall], axis=1)
    dffall = dffall[(dffall["ExperimentState"] == phase)].reset_index(drop=True)
    if phase == "Full":
        dffall['Seconds']-=26
    if phase == "Dark":
        dffall['Seconds']-=3
    if phase == "Recovery":
        dffall['Seconds']-=46
            
    df_test = dffall.copy()
    for r in dffall.iloc[:,2:].columns:
        df_temp = pd.DataFrame()
        df_temp['Time ' + r] = [0]*len(dffall)
        df_test = pd.concat([df_test,df_temp], axis = 1)
        df_test.loc[(dffall[r]>0), ['Time ' +r]] = df_test['Seconds']
    df_test= df_test.filter(regex="Time .*")
    df_test2 = pd.concat([dffall['ExperimentState'],df_test], axis = 1)
    dfuu = pd.melt(df_test2, id_vars=['ExperimentState'])
    dfuu= dfuu.replace(0.0, np.nan, regex=True)
    
    return dfuu

# def rastergraph(dfexpt):   #obsolete
#     import pandas as pd
    
        
#     phase= ["Dark", "Full", 'Recovery']
#     dfn = pd.DataFrame()
#     for n in phase:
#         dta= dfexpt[(dfexpt['ExperimentState'] == n)].copy()
#         dftot = pd.DataFrame()
#         if n == "Full":
#             dta['Seconds'] -= 23
#         if n == "Recovery":
#             dta['Seconds'] -= 46       
#         dftot = pd.concat([dta['Seconds'], dta.filter(regex="Fall.*")], axis = 1).reset_index(drop=True)
    
#         df_test = dftot.copy()
#         dfuu = pd.DataFrame()
#         for r in dftot.iloc[:,2:].columns:
#             df_temp = pd.DataFrame()
#             df_temp['Time ' + r] = [0]*len(dftot)
#             df_test = pd.concat([df_test,df_temp], axis = 1)
#             df_test.loc[(dftot[r]>0), ['Time ' +r]] = df_test['Seconds']
#             df_test2= df_test.filter(regex="Time .*")
#             dfuu = pd.melt(df_test2)
#             dfuu["ExperimentState"] = n
#         dfn = pd.concat([dfn, dfuu])
#         dfu2 = dfn[dfn['value'] > 0].reset_index(drop=True)  
    
#     return dfu2

def velodabest(df, typeo, keyword):
    import pandas as pd
    #typeo is either WT or EXPT
    keywordnew = keyword +".*"
    
    phase = ["Dark", "Full", "Recovery"]
    fgt2b = pd.DataFrame()
    for n in phase:
        dfsed = calcgraph(df, keywordnew)
        df_ff = dfsed[(dfsed['ExperimentState']== n)] 
        fgt = pd.DataFrame()
        fgt[keyword]= df_ff.iloc[:,2:].mean(axis=0)
        fgt["ExperimentState"] = n
        fgt2b = pd.concat([fgt2b, fgt])
        
    fgt2b["Type"] = typeo
    
    if any(fgt2b[keyword].isnull()):
        value = fgt2b[fgt2b[keyword].isnull()].index.tolist()[0]
        fgt2b = fgt2b.drop(index= value)
    
    return fgt2b

def maxheight(results, genre):
    import pandas as pd
    
    yonly = pd.DataFrame()
    df = results[(results['ExperimentState'] == 'Assimilation time - Dark') | (results['ExperimentState']== 'Dark')] 
    yonly = pd.concat([df['Seconds'], df.filter(regex="Y.*")], axis = 1)
    tacalc = pd.DataFrame()

    maxilst=[]
    timelst=[]

    for b in yonly.columns[0:-1]:
        maxi = yonly[b].max(axis=0)
        indx = yonly[b].idxmax()
        time = yonly.loc[indx,"Seconds"]
        maxilst.append(maxi)
        timelst.append(time)

    tacalc['Max height '+ genre] = maxilst
    tacalc["Time to reach max height " + genre]=timelst

    return tacalc

def timespentabovemeanline(dfexpt, avgmaxheight, phase):
    import pandas as pd
    
    expt = pd.DataFrame()
    expt = dfexpt.filter(regex = "Y.*")
    expt = expt.reset_index(drop=True)
    length = len(dfexpt[(dfexpt['ExperimentState'] == 'Assimilation time - Dark') | (dfexpt['ExperimentState']== 'Dark')])
    ho2=pd.DataFrame()

    for n,k in zip(expt.columns, range(1, len(expt.columns)+1)):
        nno = pd.DataFrame()
        nno[n] = expt[n]
        nno['Time_'+ str(k)] = 0
        nno.loc[(nno[n]>=0.75*avgmaxheight),['Time_'+ str(k)]] = 1
        ho2 = pd.concat([ho2, nno], axis = 1)
        
    ce = pd.DataFrame()
    #time they spend above avgmaxheight
    ce["Time"] = ho2.filter(regex="Time.*").iloc[0:length,:].sum(axis=0)
    ce["ExperimentState"] = "Dark"
    
    ef = pd.DataFrame()
    ef["Time"] = ho2.filter(regex="Time.*").iloc[length:length+length,:].sum(axis=0)
    ef["ExperimentState"] = "Full"
    
    ef6 = pd.DataFrame()
    ef6["Time"] = ho2.filter(regex="Time.*").iloc[length+length::,:].sum(axis=0)
    ef6["ExperimentState"] = "Recovery"
    
    eef = pd.DataFrame()
    eef=pd.concat([ce,ef, ef6]).reset_index(drop=False)
    eef['Type'] = phase
    eef.iloc[:,1] = eef.iloc[:,1]*1/frames(dfexpt)
    
    return eef

def timetoreach(dfexpt, avgmaxheight, phase): #obsolete
    import pandas as pd
        
    expt = pd.DataFrame()
    expt = dfexpt.filter(regex = "Y.*")
    expt = expt.reset_index(drop=True)
    length = len(dfexpt[(dfexpt['ExperimentState'] == 'Assimilation time - Dark') | (dfexpt['ExperimentState']== 'Dark')]) #supposedly 115

    ho2=pd.DataFrame()
    ho2['Seconds'] = dfexpt.loc[:,'Seconds']
    for n,k in zip(expt.columns, range(1, len(expt.columns)+1)):
        nno = pd.DataFrame()
        nno[n] = expt[n]
        nno['Time_'+ str(k)] = 0
        nno.loc[(nno[n]>=0.75*avgmaxheight),['Time_'+ str(k)]] = 1
        ho2 = pd.concat([ho2, nno], axis = 1)
        
        hoho = ho2.filter(regex="Time.*").iloc[0:length,:] #0 - 115
        indx = hoho.idxmax()
        time = ho2.loc[indx, 'Seconds']
        
        hohoho = ho2.filter(regex="Time.*").iloc[length:length+length,:].reset_index(drop=True) # 115-230
        indx2 = hohoho.idxmax()
        time2 = ho2.loc[indx2, 'Seconds']
        
        hoho3 = ho2.filter(regex="Time.*").iloc[length+length::,:].reset_index(drop=True) #230 - end
        indx3 = hoho3.idxmax()
        time3 = ho2.loc[indx3, 'Seconds']
        
    ce = pd.DataFrame()
    ce["Time"] = time.reset_index(drop=True)
    ce["ExperimentState"] = "Dark"
    
    ef = pd.DataFrame()
    ef["Time"] = time2.reset_index(drop=True)
    ef["ExperimentState"] = "Full"  
    
    eef = pd.DataFrame()
    eef=pd.concat([ce,ef]).reset_index(drop=True)
    
    #eef['Time'] = eef['Time'].replace({'0':np.nan, 0:np.nan})
    eef['Time'] = eef['Time'].replace({'0':23, 0:23})
        
    ef6 = pd.DataFrame()
    ef6["Time"] = time3.reset_index(drop=True)
    ef6["ExperimentState"] = "Recovery"
    eef = pd.concat([eef, ef6]).reset_index(drop=False)
    eef['Time'] = eef['Time'].replace({'0':46, 0:46})    
    eef['Type'] = phase
    
    return eef

def speedcalc(df, fps):
    import pandas as pd
    import numpy as np
    
    indices = list(range(1,len(df.columns),2))
    rows = list(range(0,len(df)-1))
    df_disp = pd.DataFrame()
    
    for i,kk in zip(indices, range(1,len(indices)+1)):  #parsing through each object
        displacement_list =[]
        temp = pd.DataFrame()
        k = str(kk)
        naming = df.iloc[:,i].name#series name  
        nama = naming.split(" ")[0] 
        
        for ii in rows: #parsing through each line in column
            x1_D = df.iloc[ii,i] #0,1
            y1_D = df.iloc[ii,i+1] #0,2
            x2_D = df.iloc[ii+1,i] #1,1
            y2_D = df.iloc[ii+1,i+1] #1,2
            displacement = abs((((x2_D-x1_D)**2) + ((y2_D-y1_D)**2))**0.5)
            displacement_list.append(displacement)
        temp[nama + " Velocity_" + k]= displacement_list
        df_disp = pd.concat([df_disp, temp], axis=1).reset_index(drop=True)

        
    ca = 1/fps

    df_disp.iloc[:,:] = df_disp.iloc[:,:]/ca      
    
    df3 = pd.DataFrame([[np.nan] * len(df_disp.columns)], columns=df_disp.columns)
    df2 = pd.concat([df3, df_disp], ignore_index=True)
    #df2 = df3.concat(df_disp, ignore_index=True)
    
    df2['Seconds'] = df['Seconds'].reset_index(drop=True)
    return df2

def avgmean(df, phase, fps):
    import pandas as pd
    df9 = pd.DataFrame()
    dfd = pd.DataFrame()
    
    Dark_phase_X_Y= df[(df['ExperimentState']== phase)].drop(df.columns[[1]],axis = 1)
    df_speed_D = speedcalc(Dark_phase_X_Y, fps)    
    df9 = df_speed_D.iloc[:,0:-1]
    dfd["Mean"]= df9.mean(axis=1)

    return dfd       
         
def falldbest(df, name):
    import pandas as pd
    
    phase = ['Dark', 'Full', 'Recovery']
    awt2b = pd.DataFrame()
    for n in phase:
        awt=pd.DataFrame()
        df_0 = df[(df['ExperimentState']== n)] 
        filtered3 = (df_0.filter(regex="Fall.*").sum(axis=0))
        awt['Falls']=filtered3
        awt['ExperimentState'] = n
        awt = awt.reset_index()
        awt2b = pd.concat([awt2b, awt], axis = 0)


    awt2b['Type'] = name

    return awt2b
