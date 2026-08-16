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
        fallo.loc[(fallo['Diff_'+ kk ]<-3.17),[fa + ' Fall_'+ kk]] = 1   # threshold derived from confusion matrix on 2025-12-26
        # fallo['displacement_'+ kk]=0
        # fallo.loc[(fallo['Diff_'+ kk]<-3.17),['displacement_'+ kk]] = fallo['Diff_'+kk]   # threshold derived from confusion matrix on 2025-12-26
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
            dfp.loc[(ss[n]<2.588),[nama + ' Pausecount_' + k]]= 1  # threshold derived from confusion matrix on 2025-12-26
    
    
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
            displacement = abs((((x2_D-x1_D)**2) + ((y2_D-y1_D)**2))**0.5)   #is actually speed
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
    
    Dark_phase_X_Y = separation(df, "Dark").iloc[:-4]
    Light_phase_X_Y = separation(df, "Full").iloc[:-4]
    Rec_phase = separation(df, "Recovery")
    
    fps = frames(df)
    
    #falling occurence
    dff_dark = df[(df['ExperimentState'] == 'Assimilation time - Dark') | (df['ExperimentState']== 'Dark')].iloc[:-4]
    dff_light = df[(df['ExperimentState'] == 'Assimilation time - Full') | (df['ExperimentState']== 'Full')].iloc[:-4]
    dff_rec = df[(df['ExperimentState']== 'Recovery')] 
    consolidateddf = pd.concat([dff_dark, dff_light, dff_rec], axis=0).reset_index(drop=True) #added this line to remove the unncessary second after each phase
    
    dff_d=fallso(dff_dark)
    dff_l=fallso(dff_light)
    dff_r = fallso(dff_rec)
    
    dfftot2 = pd.DataFrame()
    dfftot2 = pd.concat([dff_d, dff_l, dff_r])
    dfftot3 = dfftot2.filter(regex = "Fall.*").reset_index(drop=True)
    
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
    dffnew = pd.concat([dffnew, consolidateddf], axis=1)
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

def timerule(dfexpt):
    import pandas as pd
    
    number = 20.0  #how long of the dark/light phase i want to look at for
    timesdark = dfexpt[(dfexpt['ExperimentState']== 'Dark')][dfexpt[(dfexpt['ExperimentState']== 'Dark')]['Seconds'].between(3.0, float(3.0+number), inclusive = "both")]
    timesfull = dfexpt[(dfexpt['ExperimentState']== 'Full')][dfexpt[(dfexpt['ExperimentState']== 'Full')]['Seconds'].between(26.0, float(26.0+number), inclusive = "both")]
    timesrecovery = dfexpt[(dfexpt['ExperimentState']== 'Recovery')][dfexpt[(dfexpt['ExperimentState']== 'Recovery')]['Seconds'].between(46.0, float(46.0+number), inclusive = "both")]

    timedfs = pd.concat([timesdark, timesfull, timesrecovery]).reset_index(drop=True)

    return timedfs

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
        fgt[keyword]= df_ff.iloc[:,2:].mean(axis=0, skipna=True) #skips any values that has NaN so still retains other speeds since if even one frame has an error, the entire speed is recorded as nan
        fgt["ExperimentState"] = n
        fgt2b = pd.concat([fgt2b, fgt])
        
    fgt2b["Type"] = typeo
    
    # if any(fgt2b[keyword].isnull()):
    #     value = fgt2b[fgt2b[keyword].isnull()].index.tolist()[0]
    #     fgt2b = fgt2b.drop(index= value)
    
    return fgt2b

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

#max velocity of each fly after assimilation phase

def maxvelocity(df, genre):
    import pandas as pd

    phases = ["Dark", "Full"]
    df_max = pd.DataFrame()
    for phase in phases:
        df2 = df[(df['ExperimentState']== str(phase))]
        df_dark = df2.filter(regex="Velocity_.*")
        df_maxvelocity = pd.DataFrame()
        df_maxvelocity['maxvelocity'] = df_dark.max(axis=0)
        df_maxvelocity['ExperimentState'] = phase
        df_maxvelocity['Type'] = genre
        df_maxvelocity['genre'] = str(phase)+ " " + str(genre)
        df_max = pd.concat([df_max, df_maxvelocity], axis=0)
    return df_max

def displacementbetweenpauses(df, genre):
    import pandas as pd
    import numpy as np
    from statistics import mean
    
    df_dispp = pd.DataFrame()    
    phases = ["Dark", "Full"]
    df1 = boutdisplacement(df)
    valuedflist = []
    for phase in phases:
        df47 = df1[(df1['ExperimentState']== str(phase))]
        df46 = df47.filter(regex="Perioddisp_.*")
        dftest =pd.DataFrame()
        
        for n in df46.columns:
            df_list = []
            df_00=pd.DataFrame()
            df50 = df46[n]
            x = (df50.shift(1).isnull() & df50.notnull()).cumsum()
            
            for i,g in df50.groupby(x):
                h = g.dropna()
                sumh = np.sum(h)
                df_list.append(sumh)
                          
            vdflist = list(filter(lambda x: x != 0, df_list))
            valuedflist = [mean(vdflist) if len(vdflist) > 0 else []]
            data = {'avgdisplacementbetweenpause': valuedflist, 'ExperimentState': [phase], "Type": genre, 'genre': str(phase)+ " " + str(genre)}
            index = [n]
            df_00=pd.DataFrame(data, index = index)                
            dftest = pd.concat([dftest, df_00], axis =0)
        df_dispp = pd.concat([df_dispp, dftest], axis=0)
        
    df_dispp['avgdisplacementbetweenpause'] = pd.to_numeric(df_dispp['avgdisplacementbetweenpause'])
            
    return df_dispp

#how much they walk before a pause

def boutdisplacement(dfexpt):
    import pandas as pd
    import numpy as np
    
    dfr = dfexpt.iloc[:,2:]
    velp = pd.DataFrame()
    
    for v2 in range(3,len(dfr.columns),5): #change this number if you add more parameters
        velp = pd.concat([velp, dfr.iloc[:,v2], dfr.iloc[:,v2+1]], axis = 1)

    velplst = []
    gentype = []

    for n in velp.columns[::2]:
        velplst.append(n.split("_")[1])
        gentype.append(n.split(" ")[0])

    newspeed = pd.DataFrame()

    for n,k in zip(velplst, gentype):
        newspeed[k + " Perioddisp_" + n] = [np.nan]*len(velp)
        newspeed.loc[(velp[k + " Pausecount_" + n] ==0), [k + " Perioddisp_" + n]] = (velp[k + " Velocity_" + n])*0.2
    newspeed

    newspeed = pd.concat([dfexpt.iloc[:,0:2], newspeed], axis = 1)
    
    return newspeed

#straightness index calculations
def straightnessindexmeter(dft, genre):
    import pandas as pd
    import numpy as np
    
    phase = ['Dark', 'Full']

    dfstraighttotal = pd.DataFrame()

    for x in phase:
        df_x = dft[(dft['ExperimentState']== str(x))] 
        t1 = distpersec(df_x).iloc[:,2:]
        t2 = disppersec(df_x).iloc[:,2:] 
        v = t1.values/t2.values
        straightnessindex = pd.DataFrame(v, index=t1.index, columns=t1.columns).replace(np.inf, np.nan)
        sim = pd.DataFrame(data = straightnessindex.mean(axis=0), columns = ['averagestraightnessindex'])
        # sim = pd.melt(straightnessindex, var_name = "index", value_name = 'straightnessindex')
        sim['ExperimentState'] = x
        sim["Type"]= genre
        sim['genre']= str(x)+ " " + str(genre)
        dfstraighttotal = pd.concat([dfstraighttotal, sim], axis = 0)
    return dfstraighttotal.reset_index(drop=False)

#distance per sec

def distpersec (dfexpt):
    import pandas as pd
    import numpy as np
    
    dfnewt = dfexpt.iloc[::5,:].reset_index(drop=True)
    dfnewt.drop(dfnewt.filter(regex='Fall_.*|Velocity_.*|Pausecount_.*').columns, axis=1, inplace=True)

    dfnewt3 = dfnewt.iloc[:,2:].copy()
    distsec = pd.DataFrame()
    for v2 in range(0,len(dfnewt3.columns),2):
        dfnewt4= pd.DataFrame()
        #assining name
        naming = (dfnewt3.iloc[:,v2]).name
        arraynum = naming.split("_")[1]
        
        dfnewt4 = pd.concat([dfnewt3.iloc[:,v2], dfnewt3.iloc[:,v2+1]], axis = 1)
        distsec["Dist_" + str(arraynum)] = np.linalg.norm(dfnewt4.diff(axis=0), axis=1)
    distsec = pd.concat([dfnewt.iloc[:,0:2], distsec], axis = 1).reset_index(drop=True)
    
    return distsec
    

# for dispplacement per sec
def sectioneddispchunks(chunklist, dfdist):
    import pandas as pd
    import numpy as np

    sliced = pd.DataFrame()
    for nn in chunklist:  
        nnum = round(nn,1)
        sliced = pd.concat([sliced, dfdist[dfdist['Seconds'] ==nnum]], axis = 0)
        
    df_slice = pd.DataFrame()   

    test = sliced.iloc[:,1:]
    for v2 in range(0,len(test.columns),3):
        dfnewt4= pd.DataFrame()
        naming = (test.iloc[:,v2]).name
        arraynum = naming.split("_")[1]   
        dfnewt4 = pd.concat([test.iloc[:,v2], test.iloc[:,v2+1]], axis = 1)
        if sum(test.iloc[:,v2+2])==0.0: #accounting if there is a fall, do not calculate displacement for that moment
            linalg_variable = np.linalg.norm(dfnewt4.diff(axis=0), axis=1)
            if np.nansum(linalg_variable) < 1.0:  #if sum of displacement events is less than 0, do not want
                df_slice["Disp_" + str(arraynum)] = np.nan
            else:
                df_slice["Disp_" + str(arraynum)] = linalg_variable
        else:
            df_slice["Disp_" + str(arraynum)] = np.nan

    df_slice2 = df_slice.sum(axis=0).to_frame().T
    return df_slice2
    

def disppersec(dftest):
    import pandas as pd
    import numpy as np
    import math
    distancevelo  = dftest.filter(regex='X_.*|Y_.*|Fall_.*')
    dfdist = pd.concat([round(dftest['Seconds'],1), distancevelo], axis =1)
    listsecondsnumber = list(range(int(dftest['Seconds'].iloc[0]),math.floor(dftest['Seconds'].iloc[-1])))
    df_sumdisp = pd.DataFrame()

    for n in listsecondsnumber:
        arraylist = list(np.linspace(n,n+1,6))
        df_sumdisp = pd.concat([df_sumdisp, sectioneddispchunks(arraylist, dfdist)], axis = 0).reset_index(drop=True)
        
    df_sumdisp = df_sumdisp.shift(periods=1)
    tempsecondslist = dftest.iloc[::5, 0:2].reset_index(drop=True)
    df_sumdisp = pd.concat([tempsecondslist, df_sumdisp], axis = 1).reset_index(drop=True)

    return df_sumdisp

def pauseheight(dfexpt):
    import pandas as pd
    import numpy as np
    
    dfr = dfexpt.iloc[:,2:]
    velp = pd.DataFrame()
    
    for v2 in range(1,len(dfr.columns),5): #change this number if you add more parameters
        velp = pd.concat([velp, dfr.iloc[:,v2], dfr.iloc[:,v2+3]], axis = 1)

    velplst = []
    gentype = []

    for n in velp.columns[::2]:
        velplst.append(n.split("_")[1])
        gentype.append(n.split(" ")[0])

    newspeed = pd.DataFrame()

    for n,k in zip(velplst, gentype):
        newspeed[k + " Height_" + n] = [np.nan]*len(velp)
        newspeed.loc[(velp[k + " Pausecount_" + n] ==1), [k + " Height_" + n]] = velp[k + " Y_" + n]

    newspeed = pd.concat([dfexpt.iloc[:,0:2], newspeed], axis = 1)
    
    return newspeed

def bheight(dfexpt, dfwt):
    import pandas as pd
    import numpy as np
    
    df_se = velodabest(dfexpt, "Expt", "Height")
    df_sw = velodabest(dfwt, "WT", "Height")
    
    fgt6=pd.DataFrame()
    fgt6 = pd.concat([df_se, df_sw]).reset_index(drop=False)
    fgt6['genre'] = fgt6['ExperimentState'] + " " + fgt6['Type']

    return fgt6

#BOUTspeed
def bspeed(dfexpt, dfwt):
    import pandas as pd
    import numpy as np
    df_se = velodabest(dfexpt, "Expt", "BSpeed")
    df_sw = velodabest(dfwt, "WT", "BSpeed")
    
    fgt6=pd.DataFrame()
    fgt6 = pd.concat([df_se, df_sw]).reset_index(drop=False)
    fgt6['genre'] = fgt6['ExperimentState'] + " " + fgt6['Type']

    return fgt6

#BOUT AND PAUSE CALCULATIONS
def pausenumber (df1, genotype, genre):# genre = either pause or bout
    import pandas as pd
    import numpy as np    
       
    df = pd.DataFrame()  
    
    for n in df1.columns[1:]:
        if n.split("_")[0] == "w1118":
            type1 = "WT"
        if n.split("_")[0] == genotype:
            type1 = "Expt"
        tempnumber = pd.DataFrame()
        tempnumber[genre] = df1[n]
        tempnumber["Type"] = type1 #genre
        tempnumber["behavior"] = n.split("_")[1] #behavior
        tempnumber["ExperimentState"] = n.split("_")[2] #state
        tempnumber['genre'] = n.split("_")[2] + " " + type1
        tempnumber['index'] = df1[n.split("_")[0] + '_index']
        
        df = pd.concat([df, tempnumber], axis = 0)
            
    deltadf =  df[df["behavior"]== genre]

    #dfdiff = deltaversion(deltadf, genotype, genre)
    
    return deltadf

def boutspeed(dfexpt):
    import pandas as pd
    import numpy as np 
    import itertools
    
    dfr = dfexpt.iloc[:,2:]
    velp = pd.DataFrame()
    for v2 in range(3,len(dfr.columns),5): #change this number if you add more parameters
        velp = pd.concat([velp, dfr.iloc[:,v2], dfr.iloc[:,v2+1]], axis = 1)

    velplst = []
    gentype = []

    for n in velp.columns[::2]:
        velplst.append(n.split("_")[1])
        gentype.append(n.split(" ")[0])

    newspeed = pd.DataFrame()

    for n,k in zip(velplst, gentype):
        newspeed[k + " BSpeed_" + n] = [np.nan]*len(velp)
        newspeed.loc[(velp[k + " Pausecount_" + n] ==0), [k + " BSpeed_" + n]] = velp[k + " Velocity_" + n]

    newspeed = pd.concat([dfexpt.iloc[:,0:2], newspeed], axis = 1)
    
    return newspeed

def countval(data, value):  #value = pause events
    import pandas as pd
    import numpy as np 
    import itertools
    
    count = 0
    timelst =[]
    for key, group in itertools.groupby(data, lambda x: x == value ):
        groupAsList = list(group)
        if( key == True ):
            count += 1
            timed = 0.2*len(groupAsList)
            timelst.append(timed)

        
    return (count, timelst)

def behavior (dfp):
    import pandas as pd
    import numpy as np 
    
    pc = dfp.filter(regex="Pausecount_.*")
    countpause = []
    countbout = []
    pcpause = pd.DataFrame()
    pcbout = pd.DataFrame()

    for n in pc:
        counter1, pausetime = countval(pc[n], 1) #pause = 1
        counter0, bouttime = countval(pc[n], 0) #bout = 0
        
        countpause.append(counter1)
        countbout.append(counter0)
        
        pcpause = pd.concat([pcpause, pd.Series(pausetime, dtype='float64')], ignore_index = True, axis = 1)
        pcbout = pd.concat([pcbout, pd.Series(bouttime, dtype='float64')], ignore_index = True, axis = 1)

    return countpause, countbout, pcpause, pcbout

def boutanalysis(df_dark, phase) : 
    import pandas as pd #genre is either w1118, or driver line
    import numpy as np 
    
    countpause, countbout, pausedark, boutdark = behavior(df_dark)
    
    #avg paus time per fly (Mean Activity time spent per fly)
    meanpdark = pausedark.mean(axis = 0)
    meanbdark = boutdark.mean(axis = 0)
    meanevent = pd.DataFrame({"Pauses_" + phase: meanpdark, "Bouts_" + phase: meanbdark})
    #meandarkevent['index'] = genre + '_'+ meandarkevent['index'].astype(str)
    
    #time per activity (raw_marker_size=0.5 ,swarm_label= "Time spent per activity")
    pausedarkdf = pausedark.melt().drop(['variable'], axis =1).dropna(axis = "index")
    boutdarkdf = boutdark.melt().drop(['variable'], axis =1).dropna(axis = "index")
    timedarkevent = pd.DataFrame({"Pauses_" + phase: pausedarkdf['value'], "Bouts_" + phase: boutdarkdf['value']})
    #timedarkevent['index'] = genre + '_' + timedarkevent['index'].astype(str)
    
    #occurences
    countevent = pd.DataFrame({"Pauses_" + phase: countpause, "Bouts_" + phase: countbout})
    #countevent['index'] = genre + '_' + countevent['index'].astype(str)
    
    return countevent, meanevent, timedarkevent

def pausecomp(dft, genre): #genre is either w1118, or driver line
    import pandas as pd
    import numpy as np 
    
    df_dark = dft[(dft['ExperimentState']== 'Dark')]  #no longer accounting for assimilation time
    df_light = dft[(dft['ExperimentState']== 'Full')] 
    df_rec = dft[(dft['ExperimentState']== 'Recovery')]
    
    countdark, meandarkevent, timedarkevent  = boutanalysis(df_dark, "Dark")
    countlight, meanlightevent, timelightevent  = boutanalysis(df_light, "Full")
    countrec, meanrecevent, timerecevent  = boutanalysis(df_rec, "Recovery")
    
    totalmeanevent = pd.concat([meandarkevent, meanlightevent, meanrecevent], axis =1)
    totalmeanevent = totalmeanevent.add_prefix(genre + "_")
    totalmeanevent = totalmeanevent.reset_index(drop=False)
    totalmeanevent['index'] = genre + '_'+ totalmeanevent['index'].astype(str)
    totalmeanevent = totalmeanevent.rename(columns = {"index": genre + "_index"})
    # totaltimeevent = pd.concat([timedarkevent, timelightevent, timerecevent], axis =1)
    # totaltimeevent = totaltimeevent.add_prefix(genre + "_")
    
    totalnumberevent = pd.concat([countdark, countlight, countrec], axis =1)
    totalnumberevent = totalnumberevent.add_prefix(genre + "_")
    totalnumberevent = totalnumberevent.reset_index(drop=False)
    totalnumberevent['index'] = genre + '_'+ totalnumberevent['index'].astype(str)
    totalnumberevent = totalnumberevent.rename(columns = {"index": genre + "_index"})
    
    return totalmeanevent, totalnumberevent


#fallingoccurences
def fallingocc(dfexpt, dfwt):
    
    awt5 = refine(dfexpt, dfwt, "Fall")
    awt5['genre'] = awt5['ExperimentState'] + " " + awt5['Type']
    awt5['binary_fallvalue'] = 0
    awt5.loc[(awt5['Fall'] >0), ['binary_fallvalue']] = 1

    return awt5

def totalheight(dfexpt, dfwt):
    
    awt5 = refine(dfexpt, dfwt, "Y")
    awt5['genre'] = awt5['ExperimentState'] + " " + awt5['Type']

    return awt5


def refine(dfexpt, dfwt, phrase):
    import pandas as pd
    import numpy as np 
        
    dfe_dark = dfexpt[(dfexpt['ExperimentState']== 'Dark')] 
    dfe_full = dfexpt[(dfexpt['ExperimentState']== 'Full')] 
    dfw_dark = dfwt[(dfwt['ExperimentState']== 'Dark')] 
    dfw_full = dfwt[(dfwt['ExperimentState']== 'Full')] 
    
    filterword = phrase + ".*"
    
    expts = [dfe_dark, dfe_full, dfw_dark, dfw_full]
    results = []
    for e in expts:
        filtereddf = e.filter(regex=filterword)
    
        match phrase:
            case "Y":
                result = getattr(filtereddf, "mean")(axis=0)
            case "Fall":
                result = getattr(filtereddf, "sum")(axis=0)/1
                
        results.append(result)
        
    awt=pd.DataFrame()
    awt[phrase]= results[0]
    awt['ExperimentState'] = "Dark"

    awt2=pd.DataFrame()
    awt2[phrase]=results[1]
    awt2['ExperimentState'] = "Full"

    awt2b = pd.concat([awt, awt2]).reset_index()
    awt2b["Type"] = "Expt"


    awt3=pd.DataFrame()
    awt3[phrase]=results[2]
    awt3['ExperimentState'] = "Dark"

    awt4=pd.DataFrame()
    awt4[phrase]=results[3]
    awt4['ExperimentState'] = "Full"
    awt4b = pd.concat([awt3, awt4]).reset_index()
    awt4b["Type"] = "WT"
    

    awt5=pd.DataFrame()
    awt5 = pd.concat([awt2b, awt4b])
    
    return awt5

#overall speed

def ospeed(dfwt, dfexpt):
    import pandas as pd
    
    df_se = velodabest(dfexpt, "Expt", "Velocity")
    df_sw = velodabest(dfwt, "WT", "Velocity")
    
    fgt6=pd.DataFrame()
    fgt6 = pd.concat([df_se, df_sw]).reset_index(drop=False)
    fgt6['genre'] = fgt6['ExperimentState'] + " " + fgt6['Type']
    
    return fgt6

def deltaversion_deltag(df_sp, metric, dfnaming):
    import pandas as pd
    import dabest

    df6 = df_sp[(df_sp['ExperimentState'] != "Recovery") ]
    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]
           
    #dfsp_db2 = dabest.load(data = dfsp_db, x = ['ExperimentState', 'ExperimentState'], paired = "baseline", id_col="index", y = metric, delta2 = True, experiment = "Type", x1_level = ["Dark", "Full"], experiment_label = ["WT","Expt"] )
    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = ["Dark", "Full"], paired = "baseline", id_col="index" ) #if delta2 = dabest; deltaG = dabest_jck
    dfstatstest = dfsp_db2.hedges_g.statistical_tests  #change to delta_g if needed
        
    if dfstatstest['control'][0].split(" ")[1] == "WT" and dfstatstest['control'][1].split(" ")[1] == "Expt":
        #dfdiff = pd.DataFrame({"MBON": genotype, "delta_g": round(dfsp_db2.delta_g.delta_delta.difference,3), "g_bca_low": round(dfsp_db2.delta_g.delta_delta.bca_low,3), "g_bca_high": round(dfsp_db2.delta_g.delta_delta.bca_high,3)}, index = [genotype]) 
        dfdiff = pd.DataFrame({dfnaming +"_bootstrap": dfsp_db2.hedges_g.delta_delta.bootstraps_delta_delta.tolist(), dfnaming +"_deltag": round(dfsp_db2.hedges_g.delta_delta.difference,3)})
    return (dfdiff)

def deltaversion_meandiff(df_sp,metric, dfnaming): #you run this because since all the binary data is at the same dimension, no standardization is required and empirical delta delta is sufficient
    import pandas as pd
    import dabest

    df6 = df_sp[(df_sp['ExperimentState'] != "Recovery") ]
    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]

    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = ["Dark", "Full"], paired = "baseline", id_col="index" ) 
    dfstatstest = dfsp_db2.mean_diff.statistical_tests  
        
    if dfstatstest['control'][0].split(" ")[1] == "WT" and dfstatstest['control'][1].split(" ")[1] == "Expt":
        #dfdiff = pd.DataFrame({"MBON": genotype, "WT": round(dfstatstest['difference'][0],3), "Expt": round(dfstatstest['difference'][1],3), "delta_g": round(dfsp_db2.mean_diff.delta_delta.difference,3), "g_bca_low": round(dfsp_db2.delta_g.delta_delta.bca_low,3), "g_bca_high": round(dfsp_db2.delta_g.delta_delta.bca_high,3)}, index = [ngenotype]) #according to zinan, delta2 == deltag in meanddiff
        dfdiff = pd.DataFrame({dfnaming +"_bootstrap": dfsp_db2.mean_diff.delta_delta.bootstraps_delta_delta.tolist(), dfnaming +"_meandiff": round(dfsp_db2.mean_diff.delta_delta.difference,3)})

    return (dfdiff)

# def timeperiod(df, number):
#     df1= pd.DataFrame()
#     pos = int(number*5) #5fps
#     df_dark = df[(df['ExperimentState']== 'Dark')].iloc[0:pos,:] 
#     df_light = df[(df['ExperimentState']== 'Full')].iloc[0:pos,:] 
#     df_rec = df[(df['ExperimentState']== 'Recovery')].iloc[0:pos,:] 
    
#     df1 =pd.concat([df_dark, df_light, df_rec], axis = 0)

#     return df1.reset_index(drop=True)

def log2metric(df, metric): 
    import numpy as np
    import pandas as pd
    final_df = pd.DataFrame()
    for phase in ['Expt', 'WT']:
        pivot_df = pd.DataFrame()
        pivot_df['log2 ' + metric] = np.log2(df[df['genre'] == 'Full ' + phase][metric].reset_index(drop=True)
                                        / df[df['genre'] == 'Dark ' + phase][metric].reset_index(drop=True)
                                        )
        pivot_df['index'] = df[df['genre'] == 'Dark ' + phase]['index'].reset_index(drop=True)
        pivot_df['Type'] = phase
        final_df = pd.concat([final_df, pivot_df[['index', 'log2 ' + metric, 'Type']]])

    return final_df.reset_index(drop=True)

def singledelta(df, metric, dfnaming):
    import dabest
    import pandas as pd
    
    df_dbsingle = dabest.load(df, idx = ("WT", "Expt"), y = metric, x = 'Type')
    df_singledelta = pd.DataFrame({dfnaming +"_bootstrap": df_dbsingle.hedges_g.results.bootstraps[0].tolist(), dfnaming +"_hedgesg": round(float(df_dbsingle.hedges_g.results.difference),3)})
    return df_singledelta

def simplemetricratio(df, metric): 
    import pandas as pd
    final_df = pd.DataFrame()
    for phase in ['Expt', 'WT']:
        pivot_df = pd.DataFrame()
        pivot_df[metric] = (df[df['genre'] == 'Full ' + phase][metric].reset_index(drop=True)
                                        / df[df['genre'] == 'Dark ' + phase][metric].reset_index(drop=True)
                                        )
        pivot_df['index'] = df[df['genre'] == 'Dark ' + phase]['index'].reset_index(drop=True)
        pivot_df['Type'] = phase
        final_df = pd.concat([final_df, pivot_df[['index', metric, 'Type']]])

    return final_df.reset_index(drop=True)

def boutindex(df, metric): 
    import pandas as pd
    import numpy as np
    final_df = pd.DataFrame()
    for phase in ['Expt', 'WT']:
        pivot_df = pd.DataFrame()
        light_values = df[df['genre'] == 'Full ' + phase][metric].reset_index(drop=True)
        dark_values = df[df['genre'] == 'Dark ' + phase][metric].reset_index(drop=True)
        
        # Calculate bout index: (light - dark) / (light + dark)
        total_values = light_values + dark_values
        
        # Handle cases where total = 0 (both light and dark are 0)
        pivot_df[metric] = np.where(total_values > 0,
            (light_values - dark_values) / total_values, np.nan)
        
        pivot_df['index'] = df[df['genre'] == 'Dark ' + phase]['index'].reset_index(drop=True)
        pivot_df['Type'] = phase
        final_df = pd.concat([final_df, pivot_df[['index', metric, 'Type']]])

    return final_df.reset_index(drop=True)