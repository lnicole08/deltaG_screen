# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 23:17:03 2022

@author: lnico
"""

def removenans(df):
    perc = 50.0
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    df = df.dropna(axis=1, thresh=min_count)
    
    return df

def onlycolsneeded(df):
    cols = [col for col in df if col.endswith('X') or col.endswith('Y')]
    df = df.loc[:,cols]
    
    return df

def reassembly(results, results3, fps):
    import pandas as pd
    import numpy as np

    results2 = pd.DataFrame() 

    df_dark = results[(results['ExperimentState'] == 'Assimilation time - Dark') | (results['ExperimentState']== 'Dark')] 
    df_dark = df_dark.tail(fps*23)
    
    tt = results[(results['ExperimentState']== 'Assimilation time - Full')]
    gg = tt.groupby(np.arange(len(tt))//(len(tt)/(fps*3))).mean(numeric_only=True)
    df_lightassim = gg.rolling(5, min_periods = 1).mean()
    df_light = results[(results['ExperimentState']== 'Full')].tail(fps*20)        
    
    df_recovery = results[(results['ExperimentState']== 'DARK - RECOVERY PHASE')].head(fps*20)
    
    results2 = pd.concat([df_dark, df_lightassim, df_light, df_recovery]).reset_index(drop = True)    
    results2 = removenans(results2)      
    results2 = onlycolsneeded(results2)
    results3 = pd.concat([results3, results2], axis = 1).reset_index(drop = True)

    return results3

def jonnysmagic(df):
    
    test = df.loc[:,'Time'].reset_index()    
    test['Time_before'] = test["Time"].shift(1)
    test["e4"] = test["Time"] != test["Time_before"]
    test["Change"] = test["e4"].cumsum()

    return test

def fivefps(dfs, fps):
    import pandas as pd
    import numpy as np
    
    results3 = pd.DataFrame()
    results4 = pd.DataFrame()
    df_time = pd.DataFrame() 
    df_time['Seconds'] = np.arange(0,66,1/5)    
    
    for df in dfs:
        results = pd.DataFrame() 
        results2 = pd.DataFrame()  
        test = jonnysmagic(df)
        for i in range(1,test['Change'].max()+1):
            chunk = df[test['Change'] == i].head(5)
            results = pd.concat([results, chunk])
            
        results3 = reassembly(results, results3, fps)
        
    results4 = pd.concat([df_time, results3], axis=1)

    return results4

def mixedfps(dfs, fps):
    import pandas as pd
    import numpy as np
    
    df_time = pd.DataFrame() 
    df_time['Seconds'] = np.arange(0, 66, 1/fps)        
    results5 = pd.DataFrame()
    
    if fps==1: 
        adj_dfs=pd.DataFrame()
        for df in dfs:
            results3 = pd.DataFrame()
            results4 = pd.DataFrame() 
            if df.Seconds.diff().mean() < 0.8:
                test = jonnysmagic(df)
                new_df = df[df.index.isin(test.groupby(['Change'])['index'].min().values)] 
                new_df = reassembly(new_df, results3, fps)
                adj_dfs = pd.concat([adj_dfs, new_df], axis = 1).reset_index(drop=True)
            else:                
                df = reassembly(df, results3, fps)
                adj_dfs = pd.concat([adj_dfs, df], axis = 1).reset_index(drop=True)
                
    results5 = pd.concat([df_time, adj_dfs], axis=1)     
    
    return results5

def cleanup(results4, fps, driver):

    
    ly = []
    ly.extend(['Assimilation time - Dark' for i in range(fps*3)])
    ly.extend(['Dark' for i in range(fps*20)])
    ly.extend(['Assimilation time - Full' for i in range(fps*3)])
    ly.extend(['Full' for i in range(fps*20)])
    ly.extend(['Recovery' for i in range(fps*20)])


    newElements=[*range(1,1000,1)]
    results4.columns = [driver +' X' + '_' + str(newElements.pop(0)) if "X" in col else col for col in results4.columns] 

    newElements=[*range(1,1000,1)] #needs a second one
    results4.columns = [driver +' Y' + '_' + str(newElements.pop(0)) if "Y" in col else col for col in results4.columns]

    results4.insert(1, 'ExperimentState', ly)
    #pixel conversion
    results4.iloc[:,2:] = results4.iloc[:,2:]*0.14  

    #checking for dead flies
    #nnumber = int((len(results4.columns)-2)*0.5)

    #checkingfirstrow = results4.iloc[0,2:]
    #if (nnumber*0.7) <= checkingfirstrow.isnull().sum() <= nnumber*2:
    #    results4 = results4.iloc[1:,:]
    
    return results4

def trans(filename, driver, wt):
    import pandas as pd
    import os

    
    lst=[]
    dfs=[]


    for file_no, k in zip(os.listdir(filename), range(0,200)): 
        if file_no.lower().endswith(".csv") and wt not in file_no:   
            f = os.path.join(filename, file_no)
            df=pd.read_csv(f)
            lst.append(df.Seconds.diff().mean())
            dfs.append(df)


    if all(x<0.8 for x in lst) == True:
        fps = 5
    else:
        fps=1

    if fps ==5:
        df_t = fivefps(dfs, fps)

    if fps ==1:
        df_t = mixedfps(dfs, fps)

    df_t = cleanup(df_t, fps, driver)  
    
    return df_t, fps

def control(filename, wt):
    import os
    import pandas as pd
    lst=[]
    dfs=[]


    for file_no, k in zip(os.listdir(filename), range(0,200)): 
        if file_no.lower().endswith(".csv") and wt in file_no:   
            f = os.path.join(filename, file_no)
            df=pd.read_csv(f)
            lst.append(df.Seconds.diff().mean())
            dfs.append(df)

    if all(x<0.8 for x in lst) == True:
        fps = 5
    else:
        fps=1

    if fps ==5:
        df_t = fivefps(dfs, fps)

    if fps ==1:
        df_t = mixedfps(dfs, fps)

    df_t = cleanup(df_t, fps, wt)  
    
    return df_t, fps

def frames(df):
   
    time = df['Seconds'].iloc[2] - df['Seconds'].iloc[1]
    if time == 1.0:
        fps =1
    if time <1.0:
        fps =5
    
    return fps

def separation (df, phase):
    
    phase_X_Y= df[(df['ExperimentState'] == 'Assimilation time - '+ phase) | (df['ExperimentState']== phase)].drop(df.columns[[1]],axis = 1)
    
    return phase_X_Y

def fallso(df):
    import pandas as pd
    
    df0 = df.filter(regex="Y.*")
    fall2=pd.DataFrame()
    frontrow = df.iloc[:,0:2]
    
    for n,k in zip(df0.columns, range(1,len(df0.columns)+1)):
        kk = str(k)
        fa = str(n.split(" ")[0])
        fallo = pd.DataFrame()
        fallo['Diff_' + kk] = df0[n] - df0[n].shift(1)
        fallo[fa + ' Fall_'+ kk ] = 0
        fallo.loc[(fallo['Diff_'+ kk ]<-4.94),[fa + ' Fall_'+ kk]] = 1   #determinant of fall height
        # fallo['displacement_'+ kk]=0
        # fallo.loc[(fallo['Diff_'+ kk]<-4.94),['displacement_'+ kk]] = fallo['Diff_'+kk]   #determinant of fall height
        fall2 = pd.concat([fall2, fallo], axis = 1)

    fall2 = pd.concat([frontrow, fall2], axis=1)
    # fall2['Total falls per sec']=fall2.filter(regex = "Fall.*").sum(axis=1)    
    # fall2['Overall falls']=fall2['Total falls per sec'].cumsum()

    return fall2

def pausing(df):
    import pandas as pd
    
    #frontrow = df.iloc[:,0:2]
    ss = df.filter(regex="Velocity.*").reset_index(drop=True)
    dfp = pd.DataFrame()
    
    
    for n, k in zip(ss.columns, range(1,len(ss.columns)+1)):
            k = str(k)
            nama = n.split(" ")[0]
            dfp[nama + ' Pausecount_' + k] = [0]*len(ss)
            dfp.loc[(ss[n]<2.19),[nama + ' Pausecount_' + k]]= 1
    
    
    #dfp = pd.concat([frontrow, dfp], axis =1)
    return dfp

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

def generation(df, driver):
    import pandas as pd
    import numpy as np
    import re
    from natsort import index_natsorted
    
    Dark_phase_X_Y = separation(df, "Dark")
    Light_phase_X_Y = separation(df, "Full")
    Rec_phase = separation(df, "Recovery")
    
    fps = frames(df)
    
    #falling occurence
    dff_dark = df[(df['ExperimentState'] == 'Assimilation time - Dark') | (df['ExperimentState']== 'Dark')].iloc[:-4]
    dff_light = df[(df['ExperimentState'] == 'Assimilation time - Full') | (df['ExperimentState']== 'Full')].iloc[:-4]
    dff_rec = df[(df['ExperimentState']== 'Recovery')] 
    
    dff_d=fallso(dff_dark)
    dff_l=fallso(dff_light)
    dff_r = fallso(dff_rec)
    
    dfftot2 = pd.DataFrame()
    dfftot2 = pd.concat([dff_d, dff_l, dff_r])
    dfftot3 = dfftot2.filter(regex = "Fall.*")
    
    #speed
    df_speed_D =speedcalc(Dark_phase_X_Y, fps)
    df_speed_L = speedcalc(Light_phase_X_Y, fps)
    df_speed_R = speedcalc(Rec_phase, fps)
    
    df_speedtot = pd.DataFrame()
    df_speedtot = pd.concat([df_speed_D, df_speed_L, df_speed_R]).reset_index(drop=True) 
    dfst6 = df_speedtot.drop(["Seconds"], axis =1)    
    
    #pausing
    df_pause_D = pausing(df_speed_D)
    df_pause_L = pausing(df_speed_L)
    df_pause_R = pausing(df_speed_R)
    
    df_pausetot = pd.DataFrame()
    df_pausetot = pd.concat([df_pause_D, df_pause_L, df_pause_R]).reset_index(drop=True) 
    
    #total
    dffnew = pd.DataFrame()
    dffnew = pd.concat([dffnew, df], axis=1)
    dffnew = pd.concat([dffnew, dfftot3], axis=1)
    dffnew = pd.concat([dffnew, dfst6], axis=1)
    dffnew = pd.concat([dffnew, df_pausetot], axis =1)

    #name arranging
    heading2 = dffnew.iloc[:,2:].columns
    lstp2 = []
    pdf2 = pd.DataFrame()
    for n2 in range(0,len(heading2)):
        lstp2.append(int(re.search(r'(?<=_)\d+', heading2[n2]).group()))
    pdf2['Headings']=heading2
    pdf2['num'] = lstp2
    pdff2= pdf2.sort_values(by='num', key=lambda x: np.argsort(index_natsorted(pdf2["num"]))).reset_index(drop=True)
    dffn = dffnew.iloc[:,2:]
    dfr = dffn.reindex(columns = pdff2['Headings'])
    for v2 in range(2,len(dfr.columns),5): #change this number if you add more parameters
        for v1 in range(0,len(dfr)):
            if dfr.iloc[v1,v2]>= 1:
                dfr.iloc[v1,v2+1] = np.nan
    first2 = dffnew.iloc[:,0:2]
    dftotalexpt = pd.concat([first2, dfr],axis=1)
    
    #removing tracking errors
    chunk = len(dftotalexpt.iloc[:,2:].columns)/5  #change this number if you add more parameters
    #np.hsplit(dftotalexpt.iloc[:,2:],chunk)
    dfowo = pd.DataFrame()
    for n in np.hsplit(dftotalexpt.iloc[:,2:],chunk):
        #if velocity exceeds 100
        dfstp = n.filter(regex="Velocity.*")
        output = dfstp[(dfstp > 80)].count()
        
        temp = pd.DataFrame()
        hug = dfstp.iloc[0:23*fps]
        temp['Acc'] = abs(hug.diff())/(1/fps)
        output2 = temp[(temp < 0.04)].count() #give me the su6m of values of where acceleration is greater than 0.5

        
        #if there are dead flies
        yval = n.filter(regex="Y.*")
        #only checking dark side, not full or rec
        dfstp2 = yval.iloc[0:23*fps]
        output3 = dfstp2[(dfstp2 < 20)].count().values #if y pos less than 1 (if y poss less than 1 for duration of half times = dead fly)
        
        if int(output) <3 and int(output2) < int((23*fps)/6) and int(output3) < int((23*fps)/3) : # and int(output3) > int(15*half) and int(output2) < 10 int(output) < 300  
            dfowo=pd.concat([dfowo,n], axis=1)
                
    dfowo = pd.concat([dftotalexpt.iloc[:,0:2], dfowo], axis = 1)

    return dfowo

def fivesecondrule(dfexpt):
    import pandas as pd
    
    number = 20.0  #how long of the dark/light phase i want to look at for
    fivesecondsdark = dfexpt[(dfexpt['ExperimentState']== 'Dark')][dfexpt[(dfexpt['ExperimentState']== 'Dark')]['Seconds'].between(3.0, float(3.0+number), inclusive = "both")]
    fivesecondsfull = dfexpt[(dfexpt['ExperimentState']== 'Full')][dfexpt[(dfexpt['ExperimentState']== 'Full')]['Seconds'].between(26.0, float(26.0+number), inclusive = "both")]
    fivesecondsrecovery = dfexpt[(dfexpt['ExperimentState']== 'Recovery')][dfexpt[(dfexpt['ExperimentState']== 'Recovery')]['Seconds'].between(46.0, float(46.0+number), inclusive = "both")]

    fiveseconddfs = pd.concat([fivesecondsdark, fivesecondsfull, fivesecondsrecovery]).reset_index(drop=True)

    return fiveseconddfs