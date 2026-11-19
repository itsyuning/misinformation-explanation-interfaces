"""Reproduce the statistical results reported in the IDA paper."""
import json, hashlib, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, statsmodels
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
source=(ROOT/'data/final/task_responses_70.json')
raw=json.loads(source.read_text())
pool=json.loads((ROOT/'data/final/stimuli_pool_v11_twoyear_neutral.json').read_text())
items={x['id']:x for x in pool['items']}
rows=[]; timings=[]
for i,s in enumerate(raw['submissions']):
 for a in json.loads(s['answers_json']):
  item=items[a['item_id']]
  rows.append(dict(a,participant=f'P{i+1:02d}',item=a['item_id'],truth=item['truth']['label'],ai_correct=int(item['ai']['label']==item['truth']['label']),score=item['ai']['confidence'],duration=s['duration_sec'],list_id=s['list_id']))
 for e in json.loads(s['events_json']):
  if e['type']=='stage': timings.append(e)
d=pd.DataFrame(rows)
assert len(d)==1750 and not d.duplicated(['participant','item']).any()
for stage in ['initial','final']:
 assert d[stage+'_label'].isin(['TRUE','FAKE']).all()
 d[stage+'_correct']=(d[stage+'_label']==d.truth).astype(int)
 d[stage+'_align']=(d[stage+'_label']==d.ai_label).astype(int)
 d[stage+'_brier']=(d[stage+'_confidence']/100-d[stage+'_correct'])**2
d['shift']=(d.initial_label!=d.final_label).astype(int)
d['toward']=((d.initial_align==0)&(d.final_align==1)).astype(int)
d['away']=((d.initial_align==1)&(d.final_align==0)).astype(int)
d['delta_conf']=d.final_confidence-d.initial_confidence
d['zconfidence']=(d.initial_confidence-d.initial_confidence.mean())/d.initial_confidence.std(ddof=0)
a=d[d.condition!='C0'].copy()
desc=d.groupby('condition').agg(n=('shift','size'),initial_accuracy=('initial_correct','mean'),final_accuracy=('final_correct','mean'),shift=('shift','mean'),delta_confidence=('delta_conf','mean'),brier=('final_brier','mean'),toward=('toward','mean'),away=('away','mean'))
desc.to_csv(out/'table1.csv')
a.groupby(['condition','ai_correct']).agg(n=('shift','size'),final_alignment=('final_align','sum'),toward=('toward','sum'),brier=('final_brier','mean')).to_csv(out/'table2.csv')
result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__},'models':{}}
def save(): (out/'models.json').write_text(json.dumps(result,indent=2))
def contrasts(names,mu,cov,logistic):
 res=[]
 for correct in [0,1]:
  group=[]
  for c,b in [('C2','C1'),('C3','C2'),('C4','C3')]:
   v=np.zeros(len(names))
   for cond,sign in [(c,1),(b,-1)]:
    if cond=='C1':continue
    v[names.index(f'C(condition)[T.{cond}]')]+=sign
    if correct:v[names.index(f'C(condition)[T.{cond}]:ai_correct')]+=sign
   m=float(v@mu);se=float(np.sqrt(v@cov@v));p=2*norm.sf(abs(m/se))
   ci=np.array([m,m-1.96*se,m+1.96*se]);ci=np.exp(ci) if logistic else ci
   group.append(dict(ai_correct=correct,contrast=f'{c}-{b}',estimate=ci[0],lower=ci[1],upper=ci[2],p=p))
  adj=multipletests([x['p'] for x in group],method='holm')[1]
  for x,p in zip(group,adj):x['p_holm']=float(p)
  res+=group
 return res
models={
 'M1':(d,'final_correct ~ C(condition) + initial_correct + zconfidence'),
 'M2':(a,'final_correct ~ C(condition)*ai_correct + initial_correct + zconfidence'),
 'M3':(a,'final_align ~ C(condition)*ai_correct + initial_align + zconfidence'),
}
b=a.copy();b['zconfidence']=(b.initial_confidence-b.initial_confidence.mean())/b.initial_confidence.std(ddof=0)
models['M5']=(b[b.initial_align==0].copy(),'final_align ~ C(condition)*ai_correct + zconfidence')
c=a.copy();c['zscore']=(c.score-c.score.mean())/c.score.std(ddof=0)
models['M3_score']=(c,models['M3'][1]+' + zscore')
models['M3_duration']=(a[a.duration>=480].copy(),models['M3'][1])
for name,(frame,formula) in models.items():
 runs=[]
 for seed in [42,123]:
  np.random.seed(seed)
  m=BinomialBayesMixedGLM.from_formula(formula,{'participant':'0+C(participant)','item':'0+C(item)'},frame,vcp_p=1,fe_p=2)
  with warnings.catch_warnings(record=True) as ws:
   r=m.fit_vb(fit_method='BFGS',minim_opts={'gtol':1e-5,'maxiter':2000},scale_fe=False)
  op=r.optim_retvals
  runs.append({'seed':seed,'success':bool(op.success),'message':str(op.message),'iterations':int(op.nit),'gradient_max':float(np.max(np.abs(op.jac))),'elbo':float(-op.fun),'warnings':[str(w.message) for w in ws],'mean':r.fe_mean.tolist(),'sd':r.fe_sd.tolist()})
  if seed==42:
   record={'n':len(frame),'formula':formula,'terms':m.exog_names,'mean':r.fe_mean.tolist(),'sd':r.fe_sd.tolist(),'log_random_sd_mean':r.vcp_mean.tolist(),'log_random_sd_sd':r.vcp_sd.tolist()}
   if name not in ['M1','M2']:record['contrasts']=contrasts(m.exog_names,r.fe_mean,np.diag(r.fe_sd**2),True)
 record['runs']=runs;record['max_restart_mean_difference']=float(np.max(np.abs(np.array(runs[0]['mean'])-runs[1]['mean'])))
 result['models'][name]=record;save();print(name,[(r['success'],r['gradient_max']) for r in runs],flush=True)

# Crossed random intercepts: a single top-level group with separate variance components.
m=MixedLM.from_formula('final_brier ~ C(condition)*ai_correct + initial_brier',groups=np.ones(len(a)),re_formula='0',vc_formula={'participant':'0+C(participant)','item':'0+C(item)'},data=a)
with warnings.catch_warnings(record=True) as ws:
 r=m.fit(reml=False,method='lbfgs',maxiter=2000,full_output=True)
names=list(r.fe_params.index);cov=r.cov_params().loc[names,names].values
result['models']['M4']={'n':len(a),'formula':m.formula,'terms':names,'mean':r.fe_params.tolist(),'se':r.bse_fe.tolist(),'covariance':cov.tolist(),'converged':bool(r.converged),'random_variances':r.vcomp.tolist(),'residual_variance':r.scale,'warnings':[str(w.message) for w in ws],'contrasts':contrasts(names,r.fe_params.values,cov,False)}
with warnings.catch_warnings(record=True) as check_warnings:
 r_check=m.fit(reml=False,method='powell',maxiter=2000,full_output=True)
result['models']['M4']['optimizer_check']={'method':'powell','converged':bool(r_check.converged),'max_fixed_effect_difference':float(np.max(np.abs(r.fe_params-r_check.fe_params))),'loglik_difference':float(abs(r.llf-r_check.llf)),'warnings':[str(w.message) for w in check_warnings]}
result['models']['M4']['hessian_negative_definite']=bool(np.all(np.linalg.eigvalsh(m.hessian(r.params_object)[0])<0))
save();print('M4',r.converged,r.fe_params.to_dict(),flush=True)
print(desc.to_string(),flush=True)
