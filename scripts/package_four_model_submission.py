#!/usr/bin/env python3
from pathlib import Path
import csv, json, shutil, zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"submission_ensemble_five_slice_logit_stack"; PACKAGE=ROOT/"submission_ensemble_four_model_logit_stack"; OUT=ROOT/"results/full_four_model_submission"; INC_OUT=ROOT/"results/full_inception8_mip_submission"; LABELS=ROOT/"results/ants_registered_consecutive_resnet18_cbam/split.csv"; INC=ROOT/"results/cv5_inception_v3_adaptive8_mip/oof_predictions.npz"; EPS=1e-6
MODEL='''from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from .densenet3d import DenseNet121_3D
from .resnet3d import ResNet18CBAM3D
class CnnPair(nn.Module):
    def __init__(self):
        super().__init__(); self.resnet=ResNet18CBAM3D(); self.densenet=DenseNet121_3D()
    def forward(self,x): return torch.softmax(self.resnet(x),1)[:,1],torch.softmax(self.densenet(x),1)[:,1]
class InceptionMIP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=inception_v3(weights=None,aux_logits=False,num_classes=2,init_weights=True); old=self.net.Conv2d_1a_3x3; old.conv=nn.Conv2d(1,old.conv.out_channels,old.conv.kernel_size,stride=old.conv.stride,padding=old.conv.padding,bias=False)
    def forward(self,x): return torch.softmax(self.net(x),1)[:,1]
def load_models(path,inc_path,device):
    p=torch.load(path,map_location=device,weights_only=True); m=CnnPair(); m.resnet.load_state_dict(p["resnet_state_dict"]); m.densenet.load_state_dict(p["densenet_state_dict"]); m=m.to(device).eval(); q=torch.load(inc_path,map_location=device,weights_only=True); i=InceptionMIP(); i.net.load_state_dict(q["inception_state_dict"]); return m,i.to(device).eval()
'''
PRE='''import numpy as np
import torch
from scipy import ndimage
from .adaptive_localizer import adaptive_localization
TARGET_SHAPE=(16,64,64)
def prepare_registered(nifti_path,fixed_template):
    import ants
    moving=ants.image_read(str(nifti_path)); result=ants.registration(fixed=fixed_template,moving=moving,type_of_transform="Rigid",verbose=False); return np.asarray(result["warpedmovout"].numpy(),dtype=np.float64)
def _normalized(v):
    valid=np.isfinite(v)&(v>0); pos=v[valid]
    if not pos.size: raise RuntimeError("EMPTY_POSITIVE_REGISTERED_VOLUME")
    out=np.zeros(v.shape,dtype=np.float32); out[valid]=((v[valid]-float(pos.mean()))/max(float(pos.std()),1e-8)).astype(np.float32); return out
def prepare_tensor_from_registered(v,means,stds):
    best,_=adaptive_localization(v); s,e=int(best["start"]),int(best["end"]); n=_normalized(v); sl=np.stack([n[:,:,z].T for z in range(s,e+1)],0); sl=(sl-np.asarray(means,dtype=np.float32)[:,None,None])/np.asarray(stds,dtype=np.float32)[:,None,None]; r=ndimage.zoom(sl,tuple(TARGET_SHAPE[i]/sl.shape[i] for i in range(3)),order=1); return torch.from_numpy(r).unsqueeze(0).unsqueeze(0)
def prepare_inception_mip(v):
    best,_=adaptive_localization(v); c=(int(best["start"])+int(best["end"]))//2; s=max(0,min(c-4,v.shape[2]-8)); n=_normalized(v); mip=np.max(n[:,:,s:s+8],axis=2).T; r=ndimage.zoom(mip,(299/mip.shape[0],299/mip.shape[1]),order=1); return torch.from_numpy(r.astype(np.float32)).unsqueeze(0).unsqueeze(0)
'''
MAIN='''from pathlib import Path
import ants,json,numpy as np,pandas as pd,torch
from src.model import load_models
from src.preprocessing import prepare_registered,prepare_tensor_from_registered,prepare_inception_mip
from src.svm_features import extract_features
from src.svm_model import RbfSvm
DATA_ROOT=Path("/code_execution/data"); NIFTI_DIR=DATA_ROOT/"niftis"; SUBMISSION_FORMAT_PATH=DATA_ROOT/"submission_format.csv"; WRITE_SUBMISSION_PATH=Path("submission.csv"); ROOT=Path(__file__).resolve().parent
def main():
    sub=pd.read_csv(SUBMISSION_FORMAT_PATH); std=json.loads((ROOT/"assets/standardization.json").read_text()); md=json.loads((ROOT/"assets/model_metadata.json").read_text()); fixed=ants.image_read(str(ROOT/"assets/template.nii")); brain=ants.image_read(str(ROOT/"assets/brain_mask.nii.gz")).numpy()>0; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cnn,inc=load_models(ROOT/"assets/model/full_ensemble_weights.pt",ROOT/"assets/model/inception8_mip_weights.pt",dev); svm=RbfSvm(ROOT/"assets/model/morphometric_intensity_rbf_svm.npz"); st=md["logit_stack"]; coef=np.asarray(st["coef"]); out=[]
    with torch.inference_mode():
        for uid in sub["uid"]:
            v=prepare_registered(NIFTI_DIR/f"{uid}.nii.gz",fixed); t=prepare_tensor_from_registered(v,std["channel_means"],std["channel_stds"]); rp,dp=cnn(t.to(dev)); ip=inc(prepare_inception_mip(v).to(dev)); f,_=extract_features(v,brain); probs=np.asarray([float(rp.cpu()),float(dp.cpu()),svm.predict_probability(f),float(ip.cpu())]); z=np.log(np.clip(probs,1e-6,1-1e-6)/(1-np.clip(probs,1e-6,1-1e-6))); score=float(np.sum(coef*z)+st["intercept"]); out.append(float(np.clip(1/(1+np.exp(-np.clip(score,-40,40))),.005,.995)))
    sub["is_pathologic"]=out; sub.to_csv(WRITE_SUBMISSION_PATH,index=False)
if __name__=="__main__": main()
'''
def logit(p): p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p)-np.log1p(-p)
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if PACKAGE.exists(): shutil.rmtree(PACKAGE)
    shutil.copytree(BASE,PACKAGE); (PACKAGE/"src/model.py").write_text(MODEL); (PACKAGE/"src/preprocessing.py").write_text(PRE); (PACKAGE/"main.py").write_text(MAIN)
    shutil.copy2(INC_OUT/"inception8_mip_weights.pt",PACKAGE/"assets/model/inception8_mip_weights.pt")
    rows=list(csv.DictReader(LABELS.open(newline="",encoding="utf-8"))); uids=[r["uid"] for r in rows]; y=np.asarray([int(r["is_pathologic"]) for r in rows],int); n=len(y); base=np.full((n,3),np.nan); folds=np.full(n,-1)
    for fold in range(1,6):
        a=np.load(ROOT/f"results/fold{fold}_five_slice_no_gaussian_three_model_nested/predictions.npz"); by={str(u):i for i,u in enumerate(a["uid"])}
        for j,u in enumerate(uids):
            if u in by: i=by[u]; base[j]=[a["resnet"][i],a["densenet"][i],a["svm"][i]]; folds[j]=fold
    ip=np.asarray(np.load(INC)["prediction"],float); raw=np.column_stack((base,ip)); meta=LogisticRegression(C=1.0,solver="lbfgs",max_iter=5000,random_state=20260931).fit(logit(raw),y)
    md={"architecture":"resnet18_cbam_3d_densenet121_3d_rbf_svm_inception_v3_mip_logit_stack","training_scan_count":1362,"cnn_preprocessing":{"adaptive_slices":5,"gaussian_sigma":0.0,"input_shape":[16,64,64]},"inception_preprocessing":{"adaptive_slices":8,"mip":True,"input_shape":[1,299,299],"gaussian_sigma":0.0},"training":{"resnet_epochs":16,"densenet_epochs":14,"inception_epochs":16,"optimizer":"AdamW","scheduler":"CosineAnnealingLR","augmentation":"random_horizontal_flip_only"},"logit_stack":{"models":["resnet","densenet","svm","inception8_mip"],"coef":meta.coef_[0].tolist(),"intercept":float(meta.intercept_[0]),"oof_source":"five-fold leakage-controlled predictions"}}
    (PACKAGE/"assets/model_metadata.json").write_text(json.dumps(md,indent=2)+"\n"); archive=ROOT/"submission_ensemble_four_model_logit_stack.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc": z.write(p,p.relative_to(PACKAGE))
    (OUT/"metadata.json").write_text(json.dumps(md,indent=2)+"\n"); print(archive)
if __name__=="__main__": main()
