# %%
import re
import os
import pandas as pd
df = pd.read_csv("changeSlideName.csv", sep=",", encoding="utf-8")
changeByRecordNum = df.set_index("病歷號")
changeByRecordNum = changeByRecordNum.loc[:, changeByRecordNum.columns.isin([
                                                                            "UPN, c4lab"])]
changeByRecordNum["UPN, c4lab"] = changeByRecordNum.apply(
    lambda x: "A" + str(x["UPN, c4lab"]), axis=1)
changeByRecordNum


# %%
path = './'  # 這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
files = os.listdir(path)
n = 0
for i in files:
    isOut = True
    patientID = i.split("-")
    patientID[0] = patientID[0].strip("' ")
    patientID[0] = patientID[0].lower()
    patientID[0] = patientID[0].replace("左", "L")
    patientID[0] = patientID[0].replace("右", "R")
    # if(len(patientID)!=1):print(patientID[0])
    for j, r in changeByRecordNum.iterrows():
        if(type(j) == str):
            if j == patientID[0]:
                print("add "+r["UPN, c4lab"])
                oldname = path+files[n]
                newname = "../renameByUPN/"+r["UPN, c4lab"]+'.ndpi'
                os.rename(oldname, newname)
                isOut = False
                break
            elif j+"_x" == patientID[0]:
                print("add "+r["UPN, c4lab"]+"_x")
                oldname = path+files[n]
                newname = "../renameByUPN/"+r["UPN, c4lab"]+"_x"+'.ndpi'
                os.rename(oldname, newname)
                isOut = False
                break
            elif j+"_Lx" == patientID[0]:
                print("add "+r["UPN, c4lab"]+"_Lx")
                oldname = path+files[n]
                newname = "../renameByUPN/"+r["UPN, c4lab"]+"_Lx"+'.ndpi'
                os.rename(oldname, newname)
                isOut = False
                break
            elif j+"_Rx" == patientID[0]:
                print("add "+r["UPN, c4lab"]+"_Rx")
                oldname = path+files[n]
                newname = "../renameByUPN/"+r["UPN, c4lab"]+"_Rx"+'.ndpi'
                os.rename(oldname, newname)
                isOut = False
                break
    if(isOut and len(patientID) != 1):
        print("recordID "+patientID[0]+" is out of index.")
        with open("../renameByUPN/outOfIndex.txt", "a") as f:
            f.write(patientID[0]+"\n")
    n = n+1
