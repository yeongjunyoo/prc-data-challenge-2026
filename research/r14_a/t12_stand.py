# SPDX-License-Identifier: GPL-3.0-only
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
v=pd.read_parquet('research/r10_d/s1_rows_val.parquet')
G=v[(v.g==1)&(v.dsch>=1800)].copy()
G['art']=((G.y-G.dsch).abs()<=60).astype(int)
G['isn']=((G.y-G.dsch).abs()>60)&(G.y<60000)
G=G[G.art.astype(bool)|G.isn].copy()   # art vs normal only
print('val G d>=1800: n',len(G),'art',G.art.sum())
# training G rows from other months
t=pd.read_parquet('research/r14_a/lirf_dep_2025.parquet')
t['d']=(t.MVT_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds(); t['G']=t.CALLSIGN_flt.isna(); t['mon']=t.MVT_TIME_UTC_mvt.dt.month
y=t.TAXITIME_SEC_mvt
t['art']=((y-t.d).abs()<=60); t['norm']=~t.art&(y<60000)
T=t[t.G&(t.d>=1800)&(t.art|t.norm)&~t.mon.isin([1,7])].copy()
print('train G other months',len(T),'art',T.art.sum())
prior=T.art.mean()
def te(col,a=3.0):
    s=T.groupby(col).art.agg(['sum','count'])
    return ((s['sum']+a*prior)/(s['count']+a)).to_dict()
G['stand']=G.STAND_mvt.astype(str)
T['stand']=T.STAND_mvt.astype(str)
for name,keyf in [('stand',lambda x:x.stand),('stand_digit1',lambda x:x.stand.str[:1]),('stand_pre2',lambda x:x.stand.str[:2])]:
    T['k']=keyf(T); G['k']=keyf(G)
    m=te('k'); G['p_'+name]=G.k.map(m).fillna(prior)
    print(name,'AUC art vs norm:',round(roc_auc_score(G.art,G['p_'+name]),3))
G['p_pipe']=G.pa491/(G.pa491+G.pn491)
print('AUC pipeline pa/(pa+pn):',round(roc_auc_score(G.art,G.p_pipe),3))
G['lg_pipe']=np.log(G.p_pipe.clip(1e-4,1-1e-4)/(1-G.p_pipe.clip(1e-4,1-1e-4)))
for name in ['p_stand','p_stand_pre2']:
    G['lg_'+name]=np.log(G[name].clip(1e-3,1-1e-3)/(1-G[name].clip(1e-3,1-1e-3)))
# CV by day-blocks
rng=np.random.RandomState(0); folds=(pd.factorize(G.month.astype(str)+"_"+G.day.astype(str))[0]%5)
def cvfit(cols):
    p=np.zeros(len(G))
    for k in range(5):
        tr=folds!=k; te_=folds==k
        lr=LogisticRegression(C=10,max_iter=500).fit(G.loc[tr,cols],G.art[tr]); p[te_]=lr.predict_proba(G.loc[te_,cols])[:,1]
    return p
def brier(p): return ((p-G.art)**2).mean()
def ll(p): p=np.clip(p,1e-4,1-1e-4); return -(G.art*np.log(p)+(1-G.art)*np.log(1-p)).mean()
print('Brier pipe',round(brier(G.p_pipe),4),'logloss',round(ll(G.p_pipe),4))
for cols in [['lg_pipe'],['lg_pipe','lg_p_stand'],['lg_pipe','lg_p_stand_pre2'],['lg_p_stand']]:
    p=cvfit(cols); print(cols,'CV Brier',round(brier(p),4),'logloss',round(ll(p),4),'AUC',round(roc_auc_score(G.art,p),3))
# squared-error effect: use two-type value g=p*d+(1-p)*mu with mu=1100 on these rows (art vs norm only)
mu=1100.0
def sse(p): return (((p*G.dsch+(1-p)*mu)-G.y)**2).sum()/344419
print('val-ruler SSE contribution: pipe',round(sse(G.p_pipe),1))
for cols in [['lg_pipe','lg_p_stand'],['lg_pipe','lg_p_stand_pre2']]:
    p=cvfit(cols); print(cols,'SSE',round(sse(p),1),'delta vs pipe',round(sse(p)-sse(G.p_pipe),1))
