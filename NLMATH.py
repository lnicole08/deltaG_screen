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

#max velocity of each fly after assimilation phase

def maxvelocity(df, genre):
    import pandas as pd
    
    phases = ["Dark", "Full"]
    df_max = pd.DataFrame()
    for phase in phases:
        df2 = df[(df['ExperimentState']== str(phase))]
        df_dark = df2.filter(regex="Velocity_.*")
        df_maxvelocity = pd.DataFrame()
        df_maxvelocity['maxvelocity'] = df_dark.max(axis = 0)
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

def boutheight(dfexpt):
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
        newspeed.loc[(velp[k + " Pausecount_" + n] ==0), [k + " Height_" + n]] = velp[k + " Y_" + n]

    newspeed = pd.concat([dfexpt.iloc[:,0:2], newspeed], axis = 1)
    
    return newspeed

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

def disptravel (dft, light, genre):    
    import pandas as pd
    import numpy as np

    totaltravel = pd.DataFrame()

    totaldisptravelled = pd.DataFrame()
    dispdf = (dft.filter(regex="Velocity.*"))*0.2
    totaldisptravelled = dispdf.sum(axis= 0)

    lstdisp = totaldisptravelled.index.tolist()
    lstdisp = [s.replace("Velocity", "disp") for s in lstdisp]

    totaldisptravelled.index = lstdisp

    
    totaltravel['displacement'] = totaldisptravelled
    totaltravel["ExperimentState"] = light
    totaltravel["Type"] = genre
    totaltravel['genre'] = light + " " + genre
    

    return totaltravel.reset_index()

def totaldisp(dft, genre):
    import pandas as pd
    import numpy as np
    
    df_dark = dft[(dft['ExperimentState']== 'Dark')] 
    df_light = dft[(dft['ExperimentState']== 'Full')] 
    df_rec = dft[(dft['ExperimentState']== 'Recovery')]
    
    dispdark  = disptravel(df_dark, "Dark", genre)
    displight  = disptravel(df_light, "Full", genre)
    disprec = disptravel(df_rec, "Recovery", genre)
    
    totaldisp = pd.concat([dispdark, displight, disprec], axis = 0).reset_index(drop=True)
    
    return totaldisp   

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
        tempnumber['index'] = df1[n.split("_")[0] + '_index']
        
        df = pd.concat([df, tempnumber], axis = 0)
            
    deltadf =  df[df["behavior"]== genre]

    #dfdiff = deltaversion(deltadf, genotype, genre)
    
    return deltadf

def boutspeed(dfexpt):
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
        pc9 = pd.DataFrame()
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
    meandarkevent = pd.DataFrame({"Pauses_" + phase: meanpdark, "Bouts_" + phase: meanbdark})
    #meandarkevent['index'] = genre + '_'+ meandarkevent['index'].astype(str)
    
    #time per activity (raw_marker_size=0.5 ,swarm_label= "Time spent per activity")
    pausedarkdf = pausedark.melt().drop(['variable'], axis =1).dropna(axis = "index")
    boutdarkdf = boutdark.melt().drop(['variable'], axis =1).dropna(axis = "index")
    timedarkevent = pd.DataFrame({"Pauses_" + phase: pausedarkdf['value'], "Bouts_" + phase: boutdarkdf['value']})
    #timedarkevent['index'] = genre + '_' + timedarkevent['index'].astype(str)
    
    #occurences
    countevent = pd.DataFrame({"Pauses_" + phase: countpause, "Bouts_" + phase: countbout})
    #countevent['index'] = genre + '_' + countevent['index'].astype(str)
    
    return countevent, meandarkevent, timedarkevent

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
    
    awt5 = separation(dfexpt, dfwt, "Fall")
    awt5['genre'] = awt5['ExperimentState'] + " " + awt5['Type']
    awt5['value'] = 0
    awt5.loc[(awt5['Fall'] >0), ['value']] = 1

    return awt5

def totalheight(dfexpt, dfwt):
    
    awt5 = separation(dfexpt, dfwt, "Y")
    awt5['genre'] = awt5['ExperimentState'] + " " + awt5['Type']

    return awt5


def separation(dfexpt, dfwt, phrase):
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

def timetype(dfwt, dfexpt):
    import pandas as pd

    
    heightwt = maxheight(dfwt, "wt")
    avgmaxheight_wt = heightwt['Max height wt'].mean() 

    ce = timespentabovemeanline(dfexpt, avgmaxheight_wt, "Expt")
    ce2 = timespentabovemeanline(dfwt, avgmaxheight_wt, "WT")

    timehang = pd.DataFrame()
    timehang = pd.concat([ce, ce2], axis = 0)
    timehang['genre'] = timehang['ExperimentState'] + " " + timehang['Type']
    
    return timehang

#overall speed

def ospeed(dfwt, dfexpt):
    import pandas as pd
    
    df_se = velodabest(dfexpt, "Expt", "Velocity")
    df_sw = velodabest(dfwt, "WT", "Velocity")
    
    fgt6=pd.DataFrame()
    fgt6 = pd.concat([df_se, df_sw]).reset_index(drop=False)
    fgt6['genre'] = fgt6['ExperimentState'] + " " + fgt6['Type']
    
    return fgt6

def deltaversion(df_sp, genotype, metric):
    import pandas as pd
    import dabest_jck

    df6 = df_sp[(df_sp['ExperimentState'] != "Recovery") ]
    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]
           
    #dfsp_db2 = dabest_jck.load(data = dfsp_db, x = ['ExperimentState', 'ExperimentState'], paired = "baseline", id_col="index", y = metric, delta2 = True, experiment = "Type", x1_level = ["Dark", "Full"], experiment_label = ["WT","Expt"] )
    dfsp_db2 = dabest_jck.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = ["Dark", "Full"], paired = "baseline", id_col="index" ) #if delta2 = dabest; deltaG = dabest_jck
    dfstatstest = dfsp_db2.delta_g.statistical_tests
        
    if dfstatstest['control'][0].split(" ")[1] == "WT" and dfstatstest['control'][1].split(" ")[1] == "Expt":
        dfdiff = pd.DataFrame({"MBON": genotype, "WT": round(dfstatstest['difference'][0],3), "Expt": round(dfstatstest['difference'][1],3), "delta_g": round(dfsp_db2.delta_g.delta_delta.difference,3)}, index = [genotype])


    return (dfdiff)

# def timeperiod(df, number):
#     df1= pd.DataFrame()
#     pos = int(number*5) #5fps
#     df_dark = df[(df['ExperimentState']== 'Dark')].iloc[0:pos,:] 
#     df_light = df[(df['ExperimentState']== 'Full')].iloc[0:pos,:] 
#     df_rec = df[(df['ExperimentState']== 'Recovery')].iloc[0:pos,:] 
    
#     df1 =pd.concat([df_dark, df_light, df_rec], axis = 0)

#     return df1.reset_index(drop=True)