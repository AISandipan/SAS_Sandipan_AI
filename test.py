

import saspy

sas = saspy.SASsession(
    cfgfile='sascfg_personal.py',
    cfgname='metawin'
)

print(sas)