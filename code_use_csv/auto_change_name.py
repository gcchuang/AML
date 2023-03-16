# %%
import re
import os
import pandas as pd
df = pd.read_csv("changeSlideName.csv", sep=",", encoding="utf-8")
changeByName = df.set_index("Name")
changeByName = changeByName.loc[:, changeByName.columns.isin(["UPN, c4lab"])]
changeByName["UPN, c4lab"] = changeByName.apply(
    lambda x: "A" + str(x["UPN, c4lab"]), axis=1)
# changeByName


# %%
path = './'  # 這就是欲進行檔名更改的檔案路徑，路徑的斜線是為/，要留意下！
files = os.listdir(path)
n = 0
for i in files:
    isOut = True
    patientName = re.sub('[^\u4e00-\u9fa5^x^X]', '', i)
    patientName = patientName.strip("")
    patientName = patientName.lower()
    if len(patientName) != 0:
        # print(patientName)
        for j, r in changeByName.iterrows():
            if(type(j) == str):
                if j == patientName:
                    # print(patientName)
                    isOut = False
                    oldname = path+files[n]
                    newname = "../renameByUPN/"+r["UPN, c4lab"]+'.ndpi'
                    os.rename(oldname, newname)
                    break
                elif j+'x' == patientName:
                    # print(patientName)
                    oldname = path+files[n]
                    newname = "../renameByUPN/"+r["UPN, c4lab"]+"_x"+'.ndpi'
                    os.rename(oldname, newname)
                    isOut = False
                    break
                elif j+'右x' == patientName:
                    # print(patientName)
                    oldname = path+files[n]
                    newname = "../renameByUPN/"+r["UPN, c4lab"]+"_Rx"+'.ndpi'
                    os.rename(oldname, newname)
                    isOut = False
                    break
                elif j+'左x' == patientName:
                    # print(patientName)
                    oldname = path+files[n]
                    newname = "../renameByUPN/"+r["UPN, c4lab"]+"_Lx"+'.ndpi'
                    os.rename(oldname, newname)
                    isOut = False
                    break
        if(isOut):
            with open("../renameByUPN/outOfIndex.txt", "a") as f:
                f.write(patientName+"\n")
    n = n+1
