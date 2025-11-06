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
    
    phases = ["Dark", "Full", "Recovery"]
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


#fallingoccurences
def fallingocc(dfexpt, dfwt):
    
    awt5 = separation(dfexpt, dfwt, "Fall")
    awt5['genre'] = awt5['ExperimentState'] + " " + awt5['Type']
    awt5['binary_fallvalue'] = 0
    awt5.loc[(awt5['Fall'] >0), ['binary_fallvalue']] = 1

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
    dfe_rec = dfexpt[(dfexpt['ExperimentState']== 'Recovery')]
    dfw_dark = dfwt[(dfwt['ExperimentState']== 'Dark')] 
    dfw_full = dfwt[(dfwt['ExperimentState']== 'Full')] 
    dfw_rec = dfwt[(dfwt['ExperimentState']== 'Recovery')]
    
    filterword = phrase + ".*"
    
    expts = [dfe_dark, dfe_full, dfe_rec, dfw_dark, dfw_full, dfw_rec]
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
    
    awt2c=pd.DataFrame()
    awt2c[phrase]=results[2]
    awt2c['ExperimentState'] = "Recovery"

    awt2b = pd.concat([awt, awt2, awt2c]).reset_index()
    awt2b["Type"] = "Expt"


    awt3=pd.DataFrame()
    awt3[phrase]=results[3]
    awt3['ExperimentState'] = "Dark"

    awt4=pd.DataFrame()
    awt4[phrase]=results[4]
    awt4['ExperimentState'] = "Full"
    
    awt4c=pd.DataFrame()
    awt4c[phrase]=results[5]
    awt4c['ExperimentState'] = "Recovery"
    
    awt4b = pd.concat([awt3, awt4, awt4c]).reset_index()
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

def deltaversion_baseline_multistate(df_sp, metric, dfnaming, comparison_type):

    import pandas as pd
    import dabest

    # Filter data based on comparison type
    if comparison_type == 'DARK-FULL':
        df6 = df_sp[(df_sp['ExperimentState'] != "Recovery")]
        x1_level = ["Dark", "Full"]
    elif comparison_type == 'FULL-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Dark")]
        x1_level = ["Full", "Recovery"]
    elif comparison_type == 'DARK-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Full")]
        x1_level = ["Dark", "Recovery"]
    else:
        raise ValueError("comparison_type must be 'DARK-FULL', 'FULL-RECOVERY', or 'DARK-RECOVERY'")
    
    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]
           
    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = x1_level, paired = "baseline", id_col="index" )
    dfstatstest = dfsp_db2.hedges_g.results
        
    if dfstatstest['test'][1].split(" ")[1] == "Expt":
        dfdiff = pd.DataFrame({
            dfnaming + "_bootstrap": dfstatstest.loc[dfstatstest['test'].str.split(" ").str[1]  == "Expt", "bootstraps"].values[0].tolist(), 
            dfnaming + "_Hedgesg": round(dfstatstest.loc[dfstatstest['test'].str.split(" ").str[1] == "Expt", "difference"][1],3),
            "comparison_type": comparison_type
        })
    return dfdiff

def deltaversion_binarybaseline_multistate(df_sp, metric, dfnaming, comparison_type):

    import pandas as pd
    import dabest

    # Filter data based on comparison type
    if comparison_type == 'DARK-FULL':
        df6 = df_sp[(df_sp['ExperimentState'] != "Recovery")]
        x1_level = ["Dark", "Full"]
    elif comparison_type == 'FULL-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Dark")]
        x1_level = ["Full", "Recovery"]
    elif comparison_type == 'DARK-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Full")]
        x1_level = ["Dark", "Recovery"]
    else:
        raise ValueError("comparison_type must be 'DARK-FULL', 'FULL-RECOVERY', or 'DARK-RECOVERY'")

    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]

    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = x1_level, paired = "baseline", id_col="index" ) 
    dfstatstest = dfsp_db2.mean_diff.results 
        
    if dfstatstest['test'][1].split(" ")[1] == "Expt":
        dfdiff = pd.DataFrame({
            dfnaming + "_bootstrap": dfstatstest.loc[dfstatstest['test'].str.split(" ").str[1] == "Expt", "bootstraps"].values[0].tolist(), 
            dfnaming + "_Hedgesg": round(dfstatstest.loc[dfstatstest['test'].str.split(" ").str[1] == "Expt", "difference"][1],3),
            "comparison_type": comparison_type
        })

    return dfdiff

## comparison with WT states
def deltaversion_multistate(df_sp, metric, dfnaming, comparison_type):

    import pandas as pd
    import dabest

    # Filter data based on comparison type
    if comparison_type == 'DARK-FULL':
        df6 = df_sp[(df_sp['ExperimentState'] != "Recovery")]
        x1_level = ["Dark", "Full"]
    elif comparison_type == 'FULL-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Dark")]
        x1_level = ["Full", "Recovery"]
    elif comparison_type == 'DARK-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Full")]
        x1_level = ["Dark", "Recovery"]
    else:
        raise ValueError("comparison_type must be 'DARK-FULL', 'FULL-RECOVERY', or 'DARK-RECOVERY'")
    
    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]
           
    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = x1_level, paired = "baseline", id_col="index" )
    dfstatstest = dfsp_db2.hedges_g.statistical_tests
        
    if dfstatstest['control'][0].split(" ")[1] == "WT" and dfstatstest['control'][1].split(" ")[1] == "Expt":
        dfdiff = pd.DataFrame({
            dfnaming + "_bootstrap": dfsp_db2.hedges_g.delta_delta.bootstraps_delta_delta.tolist(), 
            dfnaming + "_deltag": round(dfsp_db2.hedges_g.delta_delta.difference,3),
            "comparison_type": comparison_type
        })
    return dfdiff

def deltaversion_binary_multistate(df_sp, metric, dfnaming, comparison_type):

    import pandas as pd
    import dabest

    # Filter data based on comparison type
    if comparison_type == 'DARK-FULL':
        df6 = df_sp[(df_sp['ExperimentState'] != "Recovery")]
        x1_level = ["Dark", "Full"]
    elif comparison_type == 'FULL-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Dark")]
        x1_level = ["Full", "Recovery"]
    elif comparison_type == 'DARK-RECOVERY':
        df6 = df_sp[(df_sp['ExperimentState'] != "Full")]
        x1_level = ["Dark", "Recovery"]
    else:
        raise ValueError("comparison_type must be 'DARK-FULL', 'FULL-RECOVERY', or 'DARK-RECOVERY'")

    name = []
    if any(df6[metric].isnull()):
        name = df6[df6[metric].isnull()]['index'].tolist()
    dfsp_db = df6[~df6['index'].isin(name)]

    dfsp_db2 = dabest.load(data = dfsp_db, x = ["ExperimentState", "Type"], y = metric,  delta2 = True, experiment = "Type",
                            experiment_label = ['WT', 'Expt'], x1_level = x1_level, paired = "baseline", id_col="index" ) 
    dfstatstest = dfsp_db2.mean_diff.statistical_tests  
        
    if dfstatstest['control'][0].split(" ")[1] == "WT" and dfstatstest['control'][1].split(" ")[1] == "Expt":
        dfdiff = pd.DataFrame({
            dfnaming + "_bootstrap": dfsp_db2.mean_diff.delta_delta.bootstraps_delta_delta.tolist(), 
            dfnaming + "_deltag": round(dfsp_db2.mean_diff.delta_delta.difference,3),
            "comparison_type": comparison_type
        })

    return dfdiff

